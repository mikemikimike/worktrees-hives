"""Single-unit lab run: allocate worktree + enforce findings (GH #80).

Owns the **one hypothesis / one agent|subagent / one worktree** CLI path:
allocate via :class:`~worktrees_hives.lab_jobs.LabJobManager`, optionally run a
bounded command inside the worktree, then require a valid findings pair
(``findings.json`` + ``findings.md``).

Does **not** own batch fan-out (#83), aggregate tables (#16), findings schema
(#82), or job-store internals (#85).

**Never merges.** No merge APIs; optional ``--command`` is denylisted for
``gh pr merge`` / GraphQL merge / bare force-push (``--force-with-lease`` ok).
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from worktrees_hives.errors import FindingsValidationError, PolicyError
from worktrees_hives.findings import AgentRole, FindingsReport, load_findings_pair
from worktrees_hives.lab_jobs import LabJob, LabJobError, LabJobManager

if TYPE_CHECKING:
    from collections.abc import Sequence

FINDINGS_JSON_NAME = "findings.json"
FINDINGS_MD_NAME = "findings.md"

_MERGE_TEXT_RE = re.compile(
    r"(?ix)"
    r"\bgh\s+pr\s+merge\b"
    r"|\bmergePullRequest\b"
    r"|/repos/[^/\s]+/[^/\s]+/merges?\b"
)

# git…push… with bare force remaining after stripping --force-with-lease.
_GIT_PUSH_RE = re.compile(r"(?i)\bgit(?:\.exe)?\b[\s\S]*\bpush\b")
_FORCE_WITH_LEASE_RE = re.compile(r"(?i)--force-with-lease(?:=\S+)?")
# After --force-with-lease is stripped; allow trailing quote/punct (shell wrappers).
_BARE_FORCE_RE = re.compile(
    r"(?i)(?:^|[\s'\"])(?:--force(?![\w-])|(?<![\w-])-f(?![\w-])|-[A-Za-z]*f[A-Za-z]*)"
)


class LabRunError(LabJobError):
    """Lab run orchestration failure (findings / command execution)."""


@dataclass(frozen=True, slots=True)
class LabRunResult:
    """Outcome of one ``lab run`` unit."""

    job: LabJob
    report: FindingsReport | None
    findings_json: str
    findings_md: str
    command_exit: int | None
    ok: bool
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary for the v1 CLI envelope."""
        data: dict[str, Any] = {
            "ok": self.ok,
            "job_id": self.job.job_id,
            "hypothesis_id": self.job.hypothesis_id,
            "agent_id": self.job.agent_id,
            "role": str(self.job.role),
            "owner": self.job.owner,
            "repo": self.job.repo,
            "branch": self.job.branch,
            "worktree_path": self.job.worktree_path,
            "status": str(self.job.status),
            "findings_json": self.findings_json,
            "findings_md": self.findings_md,
            "command_exit": self.command_exit,
        }
        if self.report is not None:
            data["report_status"] = str(self.report.status)
            data["report_schema_version"] = self.report.schema_version
        if self.error_code is not None:
            data["error_code"] = self.error_code
        if self.error_message is not None:
            data["error_message"] = self.error_message
        return data


def findings_paths(worktree_path: str | Path) -> tuple[Path, Path]:
    """Return default findings pair paths under a worktree root."""
    root = Path(worktree_path)
    return root / FINDINGS_JSON_NAME, root / FINDINGS_MD_NAME


def _strip_win_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def _command_to_argv(command: str | Sequence[str]) -> list[str]:
    """Normalize command to argv; platform-aware shlex for string form."""
    if isinstance(command, str):
        # posix=False preserves Windows backslashes in unquoted paths.
        parts = shlex.split(command, posix=(os.name != "nt"))
        if os.name == "nt":
            parts = [_strip_win_quotes(p) for p in parts]
        return parts
    return list(command)


def _is_git_token(token: str) -> bool:
    base = os.path.basename(token.rstrip("/\\")).casefold()
    return base in {"git", "git.exe"}


def _is_bare_force_token(token: str) -> bool:
    """True for bare force flags; False for ``--force-with-lease`` (allowed)."""
    if token == "--force" or token.startswith("--force="):
        return True
    if token == "-f":
        return True
    # Combined short options: -fu, -ff, -vf, etc. (not --force-with-lease).
    return token.startswith("-") and not token.startswith("--") and "f" in token[1:]


def _skip_git_global_options(argv: list[str], start: int) -> int:
    """Advance index past git global options to the subcommand."""
    j = start
    while j < len(argv):
        tok = argv[j]
        if not tok.startswith("-"):
            break
        # Options that require a following argument.
        if tok in {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"}:
            j += 2
            continue
        if tok.startswith(("--git-dir=", "--work-tree=", "--namespace=", "--config-env=")):
            j += 1
            continue
        if tok.startswith("--") and "=" in tok:
            j += 1
            continue
        j += 1
    return j


def _argv_has_bare_force_push(argv: list[str]) -> bool:
    """Detect bare force-push in structured argv (path-qualified git ok)."""
    i = 0
    while i < len(argv):
        if not _is_git_token(argv[i]):
            i += 1
            continue
        j = _skip_git_global_options(argv, i + 1)
        if j < len(argv) and argv[j] == "push":
            for opt in argv[j + 1 :]:
                if _is_bare_force_token(opt):
                    return True
        i += 1
    return False


def _text_has_bare_force_push(text: str) -> bool:
    """Catch wrappers (sh -c 'git push -f') via text after stripping lease form."""
    if not _GIT_PUSH_RE.search(text):
        return False
    cleaned = _FORCE_WITH_LEASE_RE.sub(" ", text)
    return _BARE_FORCE_RE.search(cleaned) is not None


def assert_command_allowed(command: str | Sequence[str]) -> None:
    """Reject merge / bare force-push (never-merge). Raises :class:`PolicyError`."""
    text = command if isinstance(command, str) else " ".join(command)
    if _MERGE_TEXT_RE.search(text):
        raise PolicyError(
            "NEVER_MERGE",
            "lab run --command refused: merge operations are denied (never-merge policy)",
        )

    argv = _command_to_argv(command)
    if _argv_has_bare_force_push(argv) or _text_has_bare_force_push(text):
        raise PolicyError(
            "BARE_FORCE_PUSH",
            "lab run --command refused: bare force-push is denied "
            "(use --force-with-lease only when policy allows; never-merge path)",
        )


def run_lab_unit(
    manager: LabJobManager,
    *,
    owner: str,
    repo: str,
    hypothesis_id: str,
    agent_id: str,
    role: AgentRole | str,
    branch: str | None = None,
    job_id: str | None = None,
    command: str | Sequence[str] | None = None,
    command_timeout: float = 3600.0,
    teardown_on_error: bool = False,
) -> LabRunResult:
    """Allocate a lab job, optionally run a command, require findings pair.

    Parameters
    ----------
    manager:
        Job manager (worktree allocate/teardown via ``wh``).
    command:
        Optional argv string or list run with ``cwd=worktree_path`` (no shell).
        Merge / bare ``git push --force`` are rejected **before** allocate.
    command_timeout:
        Positive finite seconds for the optional command.
    teardown_on_error:
        Tear down the job if command or findings validation fails.
    """
    if command is not None:
        argv_pre = _command_to_argv(command)
        if not argv_pre:
            raise LabRunError("lab run --command is empty")
        assert_command_allowed(command)
    if (
        not isinstance(command_timeout, (int, float))
        or isinstance(command_timeout, bool)
        or not math.isfinite(float(command_timeout))
        or float(command_timeout) <= 0
    ):
        raise LabRunError("command_timeout must be a finite number greater than 0")

    job = manager.allocate(
        owner=owner,
        repo=repo,
        hypothesis_id=hypothesis_id,
        agent_id=agent_id,
        role=role,
        branch=branch,
        job_id=job_id,
    )
    wt = job.worktree_path
    if not wt:
        raise LabRunError(f"allocated job {job.job_id!r} has no worktree_path")
    jpath, mpath = findings_paths(wt)
    command_exit: int | None = None
    try:
        if command is not None:
            command_exit = _run_command(command, cwd=wt, timeout=float(command_timeout))
            if command_exit != 0:
                result = LabRunResult(
                    job=job,
                    report=None,
                    findings_json=str(jpath),
                    findings_md=str(mpath),
                    command_exit=command_exit,
                    ok=False,
                    error_code="COMMAND_FAILED",
                    error_message=f"command exited {command_exit}",
                )
                return _maybe_teardown(manager, result, teardown_on_error)

        try:
            report = load_findings_pair(jpath, mpath)
            _assert_report_matches_job(report, job)
        except FindingsValidationError as exc:
            result = LabRunResult(
                job=job,
                report=None,
                findings_json=str(jpath),
                findings_md=str(mpath),
                command_exit=command_exit,
                ok=False,
                error_code="FINDINGS_INVALID",
                error_message=str(exc),
            )
            return _maybe_teardown(manager, result, teardown_on_error)

        return LabRunResult(
            job=job,
            report=report,
            findings_json=str(jpath),
            findings_md=str(mpath),
            command_exit=command_exit,
            ok=True,
        )
    except Exception:
        if teardown_on_error:
            note = _teardown_with_note(manager, job.job_id)
            if note:
                raise LabRunError(note) from None
        raise


def _maybe_teardown(
    manager: LabJobManager, result: LabRunResult, teardown_on_error: bool
) -> LabRunResult:
    if not teardown_on_error or result.ok:
        return result
    note = _teardown_with_note(manager, result.job.job_id)
    if not note:
        return result
    msg = result.error_message or ""
    combined = f"{msg}; {note}" if msg else note
    return LabRunResult(
        job=result.job,
        report=result.report,
        findings_json=result.findings_json,
        findings_md=result.findings_md,
        command_exit=result.command_exit,
        ok=False,
        error_code=result.error_code,
        error_message=combined,
    )


def _teardown_with_note(manager: LabJobManager, job_id: str) -> str | None:
    try:
        manager.teardown(job_id, force=True)
    except LabJobError as exc:
        return f"teardown failed: {exc}"
    return None


def _assert_report_matches_job(report: FindingsReport, job: LabJob) -> None:
    """Ensure findings identity matches the allocated lab job."""
    if report.hypothesis_id != job.hypothesis_id:
        raise FindingsValidationError(
            f"findings hypothesis_id {report.hypothesis_id!r} "
            f"does not match job {job.hypothesis_id!r}"
        )
    if report.agent_id != job.agent_id:
        raise FindingsValidationError(
            f"findings agent_id {report.agent_id!r} does not match job {job.agent_id!r}"
        )
    if report.role != job.role:
        raise FindingsValidationError(
            f"findings role {report.role!r} does not match job {job.role!r}"
        )
    if job.worktree_path:
        report_wt = os.path.abspath(os.path.expanduser(report.worktree))
        job_wt = os.path.abspath(os.path.expanduser(job.worktree_path))
        if report_wt != job_wt:
            raise FindingsValidationError(
                f"findings worktree {report_wt!r} does not match job {job_wt!r}"
            )


def _run_command(
    command: str | Sequence[str],
    *,
    cwd: str,
    timeout: float,
) -> int:
    assert_command_allowed(command)
    argv = _command_to_argv(command)
    if not argv:
        raise LabRunError("lab run --command is empty")

    # New session so timeout can kill the whole process group (POSIX).
    popen_kwargs: dict[str, Any] = {
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(argv, **popen_kwargs)
    except OSError as exc:
        raise LabRunError(f"lab run --command failed to start: {exc}") from exc

    try:
        _stdout, _stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(proc)
        with contextlib.suppress(Exception):
            proc.communicate(timeout=5)
        raise LabRunError(f"lab run --command timed out after {timeout}s") from exc

    # Discard output (may contain secrets); do not decode as text.
    return int(proc.returncode if proc.returncode is not None else -1)


def _kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            proc.kill()
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        proc.kill()
