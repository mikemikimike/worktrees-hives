"""Single-unit lab run: allocate worktree + enforce findings (GH #80).

Owns the **one hypothesis / one agent|subagent / one worktree** CLI path:
allocate via :class:`~worktrees_hives.lab_jobs.LabJobManager`, optionally run a
bounded command inside the worktree, then require a valid findings pair
(``findings.json`` + ``findings.md``).

Does **not** own batch fan-out (#83), aggregate tables (#16), findings schema
(#82), or job-store internals (#85).

**Never merges.** No merge APIs; optional ``--command`` is denylisted for
``gh pr merge`` / GraphQL merge / bare force-push.
"""

from __future__ import annotations

import contextlib
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from worktrees_hives.errors import FindingsValidationError
from worktrees_hives.findings import AgentRole, FindingsReport, load_findings_pair
from worktrees_hives.lab_jobs import LabJob, LabJobError, LabJobManager

if TYPE_CHECKING:
    from collections.abc import Sequence

FINDINGS_JSON_NAME = "findings.json"
FINDINGS_MD_NAME = "findings.md"

# Reject merge / destructive force-push patterns in --command (defense in depth).
_MERGE_DENY_RE = re.compile(
    r"(?ix)"
    r"\bgh\s+pr\s+merge\b"
    r"|\bmergePullRequest\b"
    r"|/repos/[^/\s]+/[^/\s]+/merges?\b"
    r"|\bgit\s+push\s+(?:-[^\s]*\s+)*(-f|--force)(?![-\w])"
    r"|\bgit\s+push\s+--force\b"
)


class LabRunError(LabJobError):
    """Lab run orchestration failure (policy / findings / command)."""


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


def assert_command_allowed(command: str | Sequence[str]) -> None:
    """Reject merge / bare force-push command strings (never-merge)."""
    text = command if isinstance(command, str) else " ".join(command)
    if _MERGE_DENY_RE.search(text):
        raise LabRunError(
            "lab run --command refused: merge and bare force-push are denied "
            f"(never-merge policy): {text!r}"
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
    command_timeout: float | None = 3600.0,
    validate_findings: bool = True,
    teardown_on_error: bool = False,
) -> LabRunResult:
    """Allocate a lab job, optionally run a command, require findings pair.

    Parameters
    ----------
    manager:
        Job manager (worktree allocate/teardown via ``wh``).
    command:
        Optional shell string or argv list run with ``cwd=worktree_path``.
        Merge / bare ``git push --force`` are rejected before spawn.
    validate_findings:
        When True (default), load and validate ``findings.json`` + ``findings.md``.
    teardown_on_error:
        Tear down the job if command or findings validation fails.
    """
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
            command_exit = _run_command(command, cwd=wt, timeout=command_timeout)
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
                if teardown_on_error:
                    _safe_teardown(manager, job.job_id)
                return result

        if not validate_findings:
            return LabRunResult(
                job=job,
                report=None,
                findings_json=str(jpath),
                findings_md=str(mpath),
                command_exit=command_exit,
                ok=True,
            )

        try:
            report = load_findings_pair(jpath, mpath)
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
            if teardown_on_error:
                _safe_teardown(manager, job.job_id)
            return result

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
            _safe_teardown(manager, job.job_id)
        raise


def _run_command(
    command: str | Sequence[str],
    *,
    cwd: str,
    timeout: float | None,
) -> int:
    assert_command_allowed(command)
    # String form is shlex-split (no shell); sequence form is argv as-is.
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    if not argv:
        raise LabRunError("lab run --command is empty")
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise LabRunError(f"lab run --command timed out after {timeout}s") from exc
    except OSError as exc:
        raise LabRunError(f"lab run --command failed to start: {exc}") from exc
    if completed.stderr:
        # Diagnostics on stderr; do not dump secrets-heavy stdout.
        print(
            completed.stderr,
            end="" if completed.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )
    return int(completed.returncode)


def _safe_teardown(manager: LabJobManager, job_id: str) -> None:
    with contextlib.suppress(LabJobError):
        manager.teardown(job_id, force=True)
