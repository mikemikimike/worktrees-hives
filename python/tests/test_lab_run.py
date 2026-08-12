"""Tests for single-unit lab run (GH #80)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from worktrees_hives.cli import main
from worktrees_hives.findings import (
    AgentRole,
    Finding,
    FindingsReport,
    FindingType,
    ReportStatus,
    write_findings_pair,
)
from worktrees_hives.lab_jobs import LabJob, LabJobStatus
from worktrees_hives.lab_run import (
    LabRunError,
    assert_command_allowed,
    findings_paths,
    run_lab_unit,
)

if TYPE_CHECKING:
    from pathlib import Path

OWNER = "acme"
REPO = "example-repo"


def _job(path: str, *, hypothesis_id: str = "H-001") -> LabJob:
    now = "2026-08-12T00:00:00Z"
    return LabJob(
        job_id=f"lab-{hypothesis_id}",
        hypothesis_id=hypothesis_id,
        agent_id="grok",
        role=AgentRole.AGENT,
        owner=OWNER,
        repo=REPO,
        branch=f"lab/{hypothesis_id}",
        worktree_path=path,
        status=LabJobStatus.ALLOCATED,
        created_at=now,
        updated_at=now,
    )


def _report(worktree: str) -> FindingsReport:
    return FindingsReport(
        hypothesis_id="H-001",
        agent_id="grok",
        role=AgentRole.AGENT,
        worktree=worktree,
        status=ReportStatus.COMPLETED,
        findings=(
            Finding(type=FindingType.DISCOVERY, summary="ok"),
            Finding(type=FindingType.NULL_RESULT, summary="no merge"),
        ),
        artifacts=("findings.json", "findings.md"),
    )


class TestNeverMerge:
    def test_command_denies_gh_pr_merge(self) -> None:
        with pytest.raises(LabRunError, match="never-merge"):
            assert_command_allowed("gh pr merge 1 --squash")

    def test_command_denies_bare_force_push(self) -> None:
        with pytest.raises(LabRunError, match="never-merge"):
            assert_command_allowed(["git", "push", "--force", "origin", "main"])

    def test_cli_has_no_merge_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            main(["merge"])


class TestRunLabUnit:
    def test_happy_path_loads_findings(self, tmp_path: Path) -> None:
        wt = tmp_path / "wt"
        wt.mkdir()
        write_findings_pair(_report(str(wt)), *findings_paths(wt))
        mgr = MagicMock()
        mgr.allocate.return_value = _job(str(wt))
        result = run_lab_unit(
            mgr,
            owner=OWNER,
            repo=REPO,
            hypothesis_id="H-001",
            agent_id="grok",
            role=AgentRole.AGENT,
        )
        assert result.ok is True
        assert result.report is not None
        assert result.report.status is ReportStatus.COMPLETED
        mgr.allocate.assert_called_once()

    def test_missing_findings_fails(self, tmp_path: Path) -> None:
        wt = tmp_path / "wt"
        wt.mkdir()
        mgr = MagicMock()
        mgr.allocate.return_value = _job(str(wt))
        result = run_lab_unit(
            mgr,
            owner=OWNER,
            repo=REPO,
            hypothesis_id="H-001",
            agent_id="grok",
            role="agent",
        )
        assert result.ok is False
        assert result.error_code == "FINDINGS_INVALID"
        assert "missing" in (result.error_message or "").lower()

    def test_command_failure(self, tmp_path: Path) -> None:
        wt = tmp_path / "wt"
        wt.mkdir()
        mgr = MagicMock()
        mgr.allocate.return_value = _job(str(wt))
        result = run_lab_unit(
            mgr,
            owner=OWNER,
            repo=REPO,
            hypothesis_id="H-001",
            agent_id="grok",
            role=AgentRole.AGENT,
            command=["false"],
            teardown_on_error=True,
        )
        assert result.ok is False
        assert result.error_code == "COMMAND_FAILED"
        assert result.command_exit != 0
        mgr.teardown.assert_called_once()

    def test_merge_command_refused_before_allocate_side_effects(self, tmp_path: Path) -> None:
        """Deny list raises from run_lab_unit after allocate when command runs."""
        wt = tmp_path / "wt"
        wt.mkdir()
        mgr = MagicMock()
        mgr.allocate.return_value = _job(str(wt))
        with pytest.raises(LabRunError, match="never-merge"):
            run_lab_unit(
                mgr,
                owner=OWNER,
                repo=REPO,
                hypothesis_id="H-001",
                agent_id="grok",
                role=AgentRole.AGENT,
                command="gh pr merge 99",
            )


class TestCliLabRun:
    def test_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["lab", "run", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "hypothesis" in out.lower()
        assert "findings" in out.lower()

    def test_json_envelope_success(self, tmp_path: Path, monkeypatch, capsys) -> None:
        wt = tmp_path / "wt"
        wt.mkdir()
        write_findings_pair(_report(str(wt)), *findings_paths(wt))
        job = _job(str(wt))

        def fake_run(manager, **kwargs):
            from worktrees_hives.lab_run import LabRunResult

            return LabRunResult(
                job=job,
                report=_report(str(wt)),
                findings_json=str(wt / "findings.json"),
                findings_md=str(wt / "findings.md"),
                command_exit=None,
                ok=True,
            )

        monkeypatch.setattr("worktrees_hives.cli.run_lab_unit", fake_run)
        monkeypatch.setattr("worktrees_hives.cli.WhClient", MagicMock)
        monkeypatch.setattr(
            "worktrees_hives.cli.LabJobManager",
            lambda *a, **k: MagicMock(),
        )
        code = main(
            [
                "--json",
                "lab",
                "run",
                "--owner",
                OWNER,
                "--repo",
                REPO,
                "--hypothesis-id",
                "H-001",
                "--agent-id",
                "grok",
            ]
        )
        assert code == 0
        env = json.loads(capsys.readouterr().out)
        assert env["ok"] is True
        assert env["schema_version"] == 1
        assert env["command"] == "lab.run"
        assert env["data"]["job_id"] == job.job_id

    def test_json_envelope_findings_failure(self, tmp_path: Path, monkeypatch, capsys) -> None:
        job = _job(str(tmp_path / "wt"))

        def fake_run(manager, **kwargs):
            from worktrees_hives.lab_run import LabRunResult

            return LabRunResult(
                job=job,
                report=None,
                findings_json=str(tmp_path / "findings.json"),
                findings_md=str(tmp_path / "findings.md"),
                command_exit=None,
                ok=False,
                error_code="FINDINGS_INVALID",
                error_message="findings JSON missing",
            )

        monkeypatch.setattr("worktrees_hives.cli.run_lab_unit", fake_run)
        monkeypatch.setattr("worktrees_hives.cli.WhClient", MagicMock)
        monkeypatch.setattr(
            "worktrees_hives.cli.LabJobManager",
            lambda *a, **k: MagicMock(),
        )
        code = main(
            [
                "--json",
                "lab",
                "run",
                "--owner",
                OWNER,
                "--repo",
                REPO,
                "--hypothesis-id",
                "H-001",
                "--agent-id",
                "grok",
            ]
        )
        assert code == 1
        env = json.loads(capsys.readouterr().out)
        assert env["ok"] is False
        assert env["error"]["code"] == "FINDINGS_INVALID"
