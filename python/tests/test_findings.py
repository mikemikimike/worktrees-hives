"""Unit tests for lab findings report schema (GH #82)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from worktrees_hives.errors import FindingsValidationError
from worktrees_hives.findings import (
    FINDINGS_SCHEMA_VERSION,
    REQUIRED_MD_SECTIONS,
    AgentRole,
    Finding,
    FindingsReport,
    FindingType,
    ReportStatus,
    empty_findings_markdown_template,
    load_findings_pair,
    parse_findings_json,
    validate_findings_markdown,
    write_findings_pair,
)

if TYPE_CHECKING:
    from pathlib import Path


def _valid_report(**overrides: object) -> FindingsReport:
    base = dict(
        hypothesis_id="H-001",
        agent_id="grok-lab",
        role=AgentRole.AGENT,
        worktree="/tmp/wt/H-001",
        status=ReportStatus.COMPLETED,
        findings=(
            Finding(
                type=FindingType.DISCOVERY,
                summary="Path sandbox holds",
                detail="No write outside worktree",
                evidence=("test_paths.py",),
            ),
            Finding(
                type=FindingType.NULL_RESULT,
                summary="No merge path exercised",
            ),
        ),
        artifacts=("findings.json", "findings.md"),
        budgets={"max_fix_commits": 3},
    )
    base.update(overrides)
    return FindingsReport(**base)  # type: ignore[arg-type]


class TestFindingsReportJson:
    def test_round_trip_dict(self) -> None:
        report = _valid_report()
        again = FindingsReport.from_dict(report.to_dict())
        assert again == report
        assert again.schema_version == FINDINGS_SCHEMA_VERSION

    def test_parse_json_text(self) -> None:
        text = _valid_report().to_json()
        parsed = parse_findings_json(text)
        assert parsed.hypothesis_id == "H-001"
        assert parsed.role is AgentRole.AGENT
        assert len(parsed.findings) == 2

    def test_rejects_empty_json(self) -> None:
        with pytest.raises(FindingsValidationError, match="empty"):
            parse_findings_json("   ")

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(FindingsValidationError, match="not valid JSON"):
            parse_findings_json("{not json")

    def test_rejects_missing_required_fields(self) -> None:
        with pytest.raises(FindingsValidationError, match="hypothesis_id"):
            FindingsReport.from_dict(
                {
                    "agent_id": "a",
                    "role": "agent",
                    "worktree": "/w",
                    "status": "completed",
                    "findings": [],
                    "artifacts": [],
                }
            )

    def test_rejects_bad_role(self) -> None:
        raw = _valid_report().to_dict()
        raw["role"] = "human"
        with pytest.raises(FindingsValidationError, match="role"):
            FindingsReport.from_dict(raw)

    def test_rejects_bad_finding_type(self) -> None:
        raw = _valid_report().to_dict()
        raw["findings"] = [{"type": "opinion", "summary": "x", "evidence": []}]
        with pytest.raises(FindingsValidationError, match=r"finding\.type"):
            FindingsReport.from_dict(raw)

    def test_rejects_missing_findings_key(self) -> None:
        raw = _valid_report().to_dict()
        del raw["findings"]
        with pytest.raises(FindingsValidationError, match="findings is required"):
            FindingsReport.from_dict(raw)

    def test_rejects_unsupported_schema_version(self) -> None:
        raw = _valid_report().to_dict()
        raw["schema_version"] = 99
        with pytest.raises(FindingsValidationError, match="schema_version"):
            FindingsReport.from_dict(raw)

    def test_budgets_optional(self) -> None:
        report = _valid_report(budgets=None)
        d = report.to_dict()
        assert "budgets" not in d
        again = FindingsReport.from_dict(d)
        assert again.budgets is None


class TestFindingsMarkdown:
    def test_template_has_all_sections(self) -> None:
        md = empty_findings_markdown_template()
        validate_findings_markdown(md)
        for title in REQUIRED_MD_SECTIONS:
            assert title.lower() in md.lower()

    def test_rendered_markdown_validates(self) -> None:
        md = _valid_report().to_markdown()
        validate_findings_markdown(md)
        assert "Path sandbox holds" in md
        assert "Attribution" in md

    def test_rejects_empty_markdown(self) -> None:
        with pytest.raises(FindingsValidationError, match="empty"):
            validate_findings_markdown("")

    def test_rejects_missing_section(self) -> None:
        md = "# Hypothesis\n\nx\n\n# Method\n\ny\n"
        with pytest.raises(FindingsValidationError, match="missing required sections"):
            validate_findings_markdown(md)


class TestFindingsPairIO:
    def test_write_and_load_pair(self, tmp_path: Path) -> None:
        report = _valid_report()
        j = tmp_path / "findings.json"
        m = tmp_path / "findings.md"
        write_findings_pair(report, j, m)
        loaded = load_findings_pair(j, m)
        assert loaded == report
        assert j.is_file() and m.is_file()
        # JSON is parseable plain text
        assert json.loads(j.read_text())["hypothesis_id"] == "H-001"

    def test_load_fails_if_json_missing(self, tmp_path: Path) -> None:
        m = tmp_path / "findings.md"
        m.write_text(empty_findings_markdown_template(), encoding="utf-8")
        with pytest.raises(FindingsValidationError, match="JSON missing"):
            load_findings_pair(tmp_path / "nope.json", m)

    def test_load_fails_if_markdown_missing(self, tmp_path: Path) -> None:
        j = tmp_path / "findings.json"
        j.write_text(_valid_report().to_json(), encoding="utf-8")
        with pytest.raises(FindingsValidationError, match="Markdown missing"):
            load_findings_pair(j, tmp_path / "nope.md")

    def test_load_fails_if_markdown_invalid(self, tmp_path: Path) -> None:
        j = tmp_path / "findings.json"
        m = tmp_path / "findings.md"
        j.write_text(_valid_report().to_json(), encoding="utf-8")
        m.write_text("# Only Hypothesis\n\nhi\n", encoding="utf-8")
        with pytest.raises(FindingsValidationError, match="Markdown"):
            load_findings_pair(j, m)
