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
import time
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

# Shell -c payload: git … push … with any force form (free-form command path).
_SHELL_C_RE = re.compile(
    r"(?ix)\b(?:sh|bash|zsh|dash|ksh|fish)\b(?:\s+-[^\s]+)*\s+-c\s+(['\"])(.*?)\1"
)
_GIT_PUSH_FORCE_IN_PAYLOAD = re.compile(
    r"(?ix)\bgit(?:\.exe)?\b(?:\s+\S+)*\s+push\b.*?"
    r"(?:--force(?:-with-lease)?(?:=|\b)|(?<![\w-])-f\b|-[A-Za-z]*f[A-Za-z]*)"
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


def _is_force_token(token: str) -> bool:
    """True for any force-push flag in free-form ``--command`` (including lease).

    ``--force-with-lease`` is only safe via Rust ``git-safe`` with branch checks;
    bare free-form lab commands must not perform force pushes at all.
    """
    if token in {"--force", "--force-with-lease"} or token.startswith(
        ("--force=", "--force-with-lease=")
    ):
        return True
    if token == "-f":
        return True
    # Combined short options: -fu, -ff, -vf, etc.
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


def _argv_has_force_push(argv: list[str]) -> bool:
    """Detect force-push in structured argv (path-qualified git ok)."""
    i = 0
    while i < len(argv):
        if not _is_git_token(argv[i]):
            i += 1
            continue
        j = _skip_git_global_options(argv, i + 1)
        if j < len(argv) and argv[j] == "push":
            for opt in argv[j + 1 :]:
                if _is_force_token(opt):
                    return True
        i += 1
    return False


def _shell_c_payloads_have_force_push(text: str) -> bool:
    """Scan only ``sh -c '…'`` payloads (avoids cross-command false positives)."""
    for match in _SHELL_C_RE.finditer(text):
        payload = match.group(2)
        if _GIT_PUSH_FORCE_IN_PAYLOAD.search(payload):
            return True
    return False


def assert_command_allowed(command: str | Sequence[str]) -> None:
    """Reject merge / force-push in free-form ``--command`` (never-merge)."""
    text = command if isinstance(command, str) else " ".join(command)
    if _MERGE_TEXT_RE.search(text):
        raise PolicyError(
            "NEVER_MERGE",
            "lab run --command refused: merge operations are denied (never-merge policy)",
        )

    argv = _command_to_argv(command)
    if _argv_has_force_push(argv) or _shell_c_payloads_have_force_push(text):
        raise PolicyError(
            "FORCE_PUSH",
            "lab run --command refused: force-push is denied on free-form commands "
            "(use Rust git-safe with expected-branch for controlled --force-with-lease)",
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
    command_started_at: float | None = None
    try:
        if command is not None:
            # Command-backed runs must produce a fresh findings pair.
            for stale in (jpath, mpath):
                with contextlib.suppress(FileNotFoundError, OSError):
                    stale.unlink()
            command_started_at = time.time()
            try:
                command_exit = _run_command(command, cwd=wt, timeout=float(command_timeout))
            except LabRunError as exc:
                # Keep allocated job in the result for automation (timeouts, spawn errors).
                result = LabRunResult(
                    job=job,
                    report=None,
                    findings_json=str(jpath),
                    findings_md=str(mpath),
                    command_exit=None,
                    ok=False,
                    error_code="COMMAND_FAILED",
                    error_message=str(exc),
                )
                return _maybe_teardown(manager, result, teardown_on_error)

        report: FindingsReport | None = None
        findings_error: str | None = None
        try:
            report = load_findings_pair(jpath, mpath)
            _assert_report_matches_job(report, job)
            if command_started_at is not None:
                _assert_findings_fresh(jpath, mpath, command_started_at)
        except FindingsValidationError as exc:
            findings_error = str(exc)
            report = None

        if findings_error is not None:
            # Prefer COMMAND_FAILED when the command itself failed, but still
            # surface findings problems when command succeeded or was absent.
            if command_exit is not None and command_exit != 0:
                result = LabRunResult(
                    job=job,
                    report=None,
                    findings_json=str(jpath),
                    findings_md=str(mpath),
                    command_exit=command_exit,
                    ok=False,
                    error_code="COMMAND_FAILED",
                    error_message=(
                        f"command exited {command_exit}; findings also invalid: {findings_error}"
                    ),
                )
            else:
                result = LabRunResult(
                    job=job,
                    report=None,
                    findings_json=str(jpath),
                    findings_md=str(mpath),
                    command_exit=command_exit,
                    ok=False,
                    error_code="FINDINGS_INVALID",
                    error_message=findings_error,
                )
            return _maybe_teardown(manager, result, teardown_on_error)

        if command_exit is not None and command_exit != 0:
            # Command failed but findings validated (partial/failed report).
            result = LabRunResult(
                job=job,
                report=report,
                findings_json=str(jpath),
                findings_md=str(mpath),
                command_exit=command_exit,
                ok=False,
                error_code="COMMAND_FAILED",
                error_message=f"command exited {command_exit}",
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
    except Exception as exc:
        if teardown_on_error:
            torn, note = _teardown_job(manager, job.job_id)
            if note:
                raise LabRunError(f"{exc}; {note}") from exc
            _ = torn
        # Convert unexpected post-allocate errors into a failed result with job id.
        if isinstance(exc, (LabRunError, PolicyError, LabJobError, FindingsValidationError)):
            raise
        result = LabRunResult(
            job=job,
            report=None,
            findings_json=str(jpath),
            findings_md=str(mpath),
            command_exit=command_exit,
            ok=False,
            error_code="LAB_RUN_ERROR",
            error_message=str(exc),
        )
        return _maybe_teardown(manager, result, teardown_on_error)


def _maybe_teardown(
    manager: LabJobManager, result: LabRunResult, teardown_on_error: bool
) -> LabRunResult:
    if not teardown_on_error or result.ok:
        return result
    torn, note = _teardown_job(manager, result.job.job_id)
    job = torn if torn is not None else result.job
    msg = result.error_message or ""
    if note:
        msg = f"{msg}; {note}" if msg else note
    return LabRunResult(
        job=job,
        report=result.report,
        findings_json=result.findings_json,
        findings_md=result.findings_md,
        command_exit=result.command_exit,
        ok=False,
        error_code=result.error_code,
        error_message=msg or None,
    )


def _teardown_job(manager: LabJobManager, job_id: str) -> tuple[LabJob | None, str | None]:
    try:
        return manager.teardown(job_id, force=True), None
    except LabJobError as exc:
        return None, f"teardown failed: {exc}"


def _assert_findings_fresh(jpath: Path, mpath: Path, started_at: float) -> None:
    """Require both findings files to be written at/after command start."""
    # Tolerate small clock skew / coarse mtime resolution.
    floor = started_at - 1.0
    for path in (jpath, mpath):
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            raise FindingsValidationError(f"findings not refreshed after command: {path}") from exc
        if mtime < floor:
            raise FindingsValidationError(f"findings stale after command (not rewritten): {path}")


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
        report_wt = os.path.normcase(os.path.abspath(os.path.expanduser(report.worktree)))
        job_wt = os.path.normcase(os.path.abspath(os.path.expanduser(job.worktree_path)))
        if report_wt != job_wt:
            # Prefer samefile when both exist (symlink / Windows case aliases).
            try:
                if os.path.samefile(report.worktree, job.worktree_path):
                    return
            except OSError:
                pass
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

    # Discard output (secrets + memory); new session for process-group kill.
    popen_kwargs: dict[str, Any] = {
        "cwd": cwd,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(argv, **popen_kwargs)
    except OSError as exc:
        raise LabRunError(f"lab run --command failed to start: {exc}") from exc

    try:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_process_tree(proc)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
            raise LabRunError(f"lab run --command timed out after {timeout}s") from exc
        code = int(proc.returncode if proc.returncode is not None else -1)
        # Reap descendants left behind by backgrounded children after launcher exit.
        _kill_process_tree(proc)
        return code
    except BaseException:
        # Ctrl+C / unexpected exit: do not orphan the lab command.
        _kill_process_tree(proc)
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        raise


def _kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            proc.kill()
        return
    # Windows: kill process tree via taskkill when available.
    with contextlib.suppress(Exception):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            check=False,
            capture_output=True,
            timeout=10,
        )
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        proc.kill()
