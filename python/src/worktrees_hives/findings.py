"""Lab findings report contract: JSON schema + Markdown template (GH #82).

Owns validation and serialization only. CLI (`lab run` / `lab batch`) and
worktree job allocation are separate issues.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from worktrees_hives.errors import FindingsValidationError

# Findings report document schema (independent of the wh CLI envelope).
FINDINGS_SCHEMA_VERSION: int = 1

# Required Markdown section titles (ATX headings, case-insensitive match on text).
REQUIRED_MD_SECTIONS: tuple[str, ...] = (
    "Hypothesis",
    "Method",
    "Discoveries",
    "Null results",
    "Errors",
    "Evidence",
    "Attribution",
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class FindingType(StrEnum):
    """Kind of a single finding entry."""

    DISCOVERY = "discovery"
    NULL_RESULT = "null_result"
    ERROR = "error"


class AgentRole(StrEnum):
    """Who produced the report."""

    AGENT = "agent"
    SUBAGENT = "subagent"


class ReportStatus(StrEnum):
    """Overall outcome of the hypothesis run."""

    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class Finding:
    """One structured finding line item."""

    type: FindingType
    summary: str
    detail: str | None = None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-compatible dict."""
        out: dict[str, Any] = {
            "type": str(self.type),
            "summary": self.summary,
            "evidence": list(self.evidence),
        }
        if self.detail is not None:
            out["detail"] = self.detail
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Finding:
        """Parse one finding object; fail closed on bad types."""
        if not isinstance(raw, dict):
            raise FindingsValidationError(f"finding must be an object, got {type(raw).__name__}")
        ftype = raw.get("type")
        if not isinstance(ftype, str):
            raise FindingsValidationError("finding.type is required and must be a string")
        try:
            kind = FindingType(ftype)
        except ValueError as exc:
            raise FindingsValidationError(
                f"finding.type must be one of {[e.value for e in FindingType]}, got {ftype!r}"
            ) from exc
        summary = raw.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise FindingsValidationError("finding.summary is required non-empty string")
        detail = raw.get("detail")
        if detail is not None and not isinstance(detail, str):
            raise FindingsValidationError("finding.detail must be a string or omitted")
        evidence_raw = raw.get("evidence", [])
        if not isinstance(evidence_raw, list) or not all(isinstance(x, str) for x in evidence_raw):
            raise FindingsValidationError("finding.evidence must be a list of strings")
        return cls(
            type=kind,
            summary=summary.strip(),
            detail=detail,
            evidence=tuple(evidence_raw),
        )


@dataclass(frozen=True, slots=True)
class FindingsReport:
    """Versioned lab findings document (JSON side of the contract)."""

    hypothesis_id: str
    agent_id: str
    role: AgentRole
    worktree: str
    status: ReportStatus
    findings: tuple[Finding, ...] = ()
    artifacts: tuple[str, ...] = ()
    budgets: dict[str, Any] | None = None
    schema_version: int = FINDINGS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-compatible dict."""
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "hypothesis_id": self.hypothesis_id,
            "agent_id": self.agent_id,
            "role": str(self.role),
            "worktree": self.worktree,
            "status": str(self.status),
            "findings": [f.to_dict() for f in self.findings],
            "artifacts": list(self.artifacts),
        }
        if self.budgets is not None:
            out["budgets"] = self.budgets
        return out

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize as JSON text."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False) + "\n"

    def to_markdown(self) -> str:
        """Render the canonical Markdown template filled from this report."""
        discoveries = [f for f in self.findings if f.type is FindingType.DISCOVERY]
        nulls = [f for f in self.findings if f.type is FindingType.NULL_RESULT]
        errors = [f for f in self.findings if f.type is FindingType.ERROR]

        def _bullets(items: list[Finding]) -> str:
            if not items:
                return "_None._\n"
            lines: list[str] = []
            for item in items:
                line = f"- **{item.summary}**"
                if item.detail:
                    line += f" — {item.detail}"
                lines.append(line)
                for ev in item.evidence:
                    lines.append(f"  - evidence: {ev}")
            return "\n".join(lines) + "\n"

        evidence_lines: list[str] = []
        for item in self.findings:
            for ev in item.evidence:
                evidence_lines.append(f"- {ev}")
        for art in self.artifacts:
            evidence_lines.append(f"- artifact: {art}")
        evidence_body = "\n".join(evidence_lines) + "\n" if evidence_lines else "_None._\n"

        return (
            f"# Hypothesis\n\n"
            f"`{self.hypothesis_id}`\n\n"
            f"# Method\n\n"
            f"Role: `{self.role}` · Agent: `{self.agent_id}` · "
            f"Worktree: `{self.worktree}` · Status: `{self.status}`\n\n"
            f"# Discoveries\n\n"
            f"{_bullets(discoveries)}\n"
            f"# Null results\n\n"
            f"{_bullets(nulls)}\n"
            f"# Errors\n\n"
            f"{_bullets(errors)}\n"
            f"# Evidence\n\n"
            f"{evidence_body}\n"
            f"# Attribution\n\n"
            f"— {self.agent_id} ({self.role}) · worktree `{self.worktree}`\n"
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FindingsReport:
        """Parse and validate a findings JSON object (fail closed)."""
        if not isinstance(raw, dict):
            raise FindingsValidationError(f"report must be an object, got {type(raw).__name__}")

        schema_version = raw.get("schema_version", FINDINGS_SCHEMA_VERSION)
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise FindingsValidationError("schema_version must be an int")
        if schema_version != FINDINGS_SCHEMA_VERSION:
            raise FindingsValidationError(
                f"unsupported schema_version {schema_version} (expected {FINDINGS_SCHEMA_VERSION})"
            )

        hypothesis_id = _require_nonempty_str(raw, "hypothesis_id")
        agent_id = _require_nonempty_str(raw, "agent_id")
        worktree = _require_nonempty_str(raw, "worktree")

        role_raw = raw.get("role")
        if not isinstance(role_raw, str):
            raise FindingsValidationError("role is required and must be a string")
        try:
            role = AgentRole(role_raw)
        except ValueError as exc:
            raise FindingsValidationError(
                f"role must be one of {[e.value for e in AgentRole]}, got {role_raw!r}"
            ) from exc

        status_raw = raw.get("status")
        if not isinstance(status_raw, str):
            raise FindingsValidationError("status is required and must be a string")
        try:
            status = ReportStatus(status_raw)
        except ValueError as exc:
            raise FindingsValidationError(
                f"status must be one of {[e.value for e in ReportStatus]}, got {status_raw!r}"
            ) from exc

        findings_raw = raw.get("findings")
        if findings_raw is None:
            raise FindingsValidationError("findings is required (use [] if empty)")
        if not isinstance(findings_raw, list):
            raise FindingsValidationError("findings must be a list")
        findings = tuple(Finding.from_dict(item) for item in findings_raw)

        artifacts_raw = raw.get("artifacts")
        if artifacts_raw is None:
            raise FindingsValidationError("artifacts is required (use [] if empty)")
        if not isinstance(artifacts_raw, list) or not all(
            isinstance(x, str) for x in artifacts_raw
        ):
            raise FindingsValidationError("artifacts must be a list of strings")

        budgets = raw.get("budgets")
        if budgets is not None and not isinstance(budgets, dict):
            raise FindingsValidationError("budgets must be an object or omitted")

        return cls(
            hypothesis_id=hypothesis_id,
            agent_id=agent_id,
            role=role,
            worktree=worktree,
            status=status,
            findings=findings,
            artifacts=tuple(artifacts_raw),
            budgets=budgets,
            schema_version=schema_version,
        )


def parse_findings_json(text: str) -> FindingsReport:
    """Decode JSON text into a validated :class:`FindingsReport`."""
    if not text or not text.strip():
        raise FindingsValidationError("findings JSON is empty")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FindingsValidationError(f"findings JSON is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise FindingsValidationError("findings JSON root must be an object")
    return FindingsReport.from_dict(raw)


def validate_findings_markdown(text: str) -> None:
    """Fail closed if Markdown is missing required section headings."""
    if not text or not text.strip():
        raise FindingsValidationError("findings Markdown is empty")
    headings = {_normalize_heading(m.group(2)) for m in _HEADING_RE.finditer(text)}
    missing = [s for s in REQUIRED_MD_SECTIONS if _normalize_heading(s) not in headings]
    if missing:
        raise FindingsValidationError(
            "findings Markdown missing required sections: " + ", ".join(missing)
        )


def load_findings_pair(
    json_path: str | Path,
    markdown_path: str | Path,
) -> FindingsReport:
    """Load and validate both sides of the contract; fail if either is missing/invalid.

    Returns the parsed JSON report after both JSON and Markdown validate.
    """
    jpath = Path(json_path)
    mpath = Path(markdown_path)
    if not jpath.is_file():
        raise FindingsValidationError(f"findings JSON missing: {jpath}")
    if not mpath.is_file():
        raise FindingsValidationError(f"findings Markdown missing: {mpath}")
    report = parse_findings_json(jpath.read_text(encoding="utf-8"))
    validate_findings_markdown(mpath.read_text(encoding="utf-8"))
    return report


def write_findings_pair(
    report: FindingsReport,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write validated JSON + rendered Markdown for a report."""
    jpath = Path(json_path)
    mpath = Path(markdown_path)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(report.to_json(), encoding="utf-8")
    md = report.to_markdown()
    validate_findings_markdown(md)
    mpath.write_text(md, encoding="utf-8")


def empty_findings_markdown_template() -> str:
    """Return the empty Markdown skeleton with all required sections."""
    parts = [f"# {title}\n\n_TODO._\n" for title in REQUIRED_MD_SECTIONS]
    return "\n".join(parts)


def _require_nonempty_str(raw: dict[str, Any], key: str) -> str:
    val = raw.get(key)
    if not isinstance(val, str) or not val.strip():
        raise FindingsValidationError(f"{key} is required non-empty string")
    return val.strip()


def _normalize_heading(text: str) -> str:
    return " ".join(text.strip().lower().split())
