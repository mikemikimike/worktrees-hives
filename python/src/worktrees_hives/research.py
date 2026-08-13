"""Versioned Research Hive hypothesis and experiment contract (GitHub #92).

This module owns the pre-execution research domain document only. It does not
run experiments, evaluate outcomes, assemble artifacts, expose CLI behavior, or
cross the Rust safety boundary.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from worktrees_hives.errors import ResearchValidationError

# Domain-document version, independent of the outer Python/Rust JSON envelope.
RESEARCH_CONTRACT_SCHEMA_VERSION: int = 1

_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "research_id",
    "question",
    "hypothesis",
    "independent_variable",
    "dependent_metrics",
    "baselines",
    "arms",
    "acceptance_criteria",
    "failure_criteria",
    "seed_policy",
    "repo",
    "ref",
    "artifact_expectations",
)

_KNOWN_FIELDS: frozenset[str] = frozenset(
    (*_REQUIRED_FIELDS, "null_hypothesis", "resource_budget", "split_policy")
)


class ResearchOutcome(StrEnum):
    """Allowed conclusions for a later Research Hive result document."""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ResearchContract:
    """Immutable, versioned description of an experiment before execution."""

    research_id: str
    question: str
    hypothesis: str
    independent_variable: str
    dependent_metrics: tuple[str, ...]
    baselines: tuple[str, ...]
    arms: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    failure_criteria: tuple[str, ...]
    seed_policy: Mapping[str, object]
    repo: str
    ref: str
    artifact_expectations: tuple[str, ...]
    null_hypothesis: str | None = None
    resource_budget: Mapping[str, object] | None = None
    split_policy: Mapping[str, object] | None = None
    extensions: Mapping[str, object] = field(default_factory=dict, repr=False)
    schema_version: int = RESEARCH_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate direct construction and defensively freeze all collections."""
        if type(self.schema_version) is not int:
            raise ResearchValidationError("schema_version must be an int")
        if self.schema_version != RESEARCH_CONTRACT_SCHEMA_VERSION:
            raise ResearchValidationError(
                f"unsupported schema_version {self.schema_version} "
                f"(expected {RESEARCH_CONTRACT_SCHEMA_VERSION})"
            )

        for name in (
            "research_id",
            "question",
            "hypothesis",
            "independent_variable",
            "repo",
            "ref",
        ):
            object.__setattr__(self, name, _require_nonempty_string(getattr(self, name), name))

        if self.null_hypothesis is not None and not isinstance(self.null_hypothesis, str):
            raise ResearchValidationError("null_hypothesis must be a string or omitted")

        collection_rules = (
            ("dependent_metrics", True),
            ("baselines", False),
            ("arms", False),
            ("acceptance_criteria", True),
            ("failure_criteria", False),
            ("artifact_expectations", False),
        )
        for name, require_nonempty in collection_rules:
            object.__setattr__(
                self,
                name,
                _freeze_string_array(
                    getattr(self, name),
                    name,
                    require_nonempty=require_nonempty,
                ),
            )

        object.__setattr__(
            self,
            "seed_policy",
            _freeze_json_object(self.seed_policy, "seed_policy"),
        )
        if self.resource_budget is not None:
            object.__setattr__(
                self,
                "resource_budget",
                _freeze_json_object(self.resource_budget, "resource_budget"),
            )
        if self.split_policy is not None:
            object.__setattr__(
                self,
                "split_policy",
                _freeze_json_object(self.split_policy, "split_policy"),
            )

        extensions = _freeze_json_object(self.extensions, "extensions")
        collisions = sorted(set(extensions).intersection(_KNOWN_FIELDS))
        if collisions:
            raise ResearchValidationError(
                "extensions cannot shadow known fields: " + ", ".join(collisions)
            )
        object.__setattr__(self, "extensions", extensions)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible object, preserving additive fields."""
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "research_id": self.research_id,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "independent_variable": self.independent_variable,
            "dependent_metrics": list(self.dependent_metrics),
            "baselines": list(self.baselines),
            "arms": list(self.arms),
            "acceptance_criteria": list(self.acceptance_criteria),
            "failure_criteria": list(self.failure_criteria),
            "seed_policy": _thaw_json_object(self.seed_policy),
            "repo": self.repo,
            "ref": self.ref,
            "artifact_expectations": list(self.artifact_expectations),
        }
        if self.null_hypothesis is not None:
            out["null_hypothesis"] = self.null_hypothesis
        if self.resource_budget is not None:
            out["resource_budget"] = _thaw_json_object(self.resource_budget)
        if self.split_policy is not None:
            out["split_policy"] = _thaw_json_object(self.split_policy)
        for name, value in self.extensions.items():
            out[name] = _thaw_json_value(value)
        return out

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize as JSON text for the domain contract."""
        try:
            return (
                json.dumps(
                    self.to_dict(),
                    indent=indent,
                    sort_keys=False,
                    allow_nan=False,
                )
                + "\n"
            )
        except (TypeError, ValueError) as exc:
            raise ResearchValidationError(f"cannot serialize contract to JSON: {exc}") from exc

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ResearchContract:
        """Parse and validate a decoded research-contract JSON object."""
        if not isinstance(raw, dict):
            raise ResearchValidationError(f"contract must be an object, got {type(raw).__name__}")
        if not all(isinstance(name, str) for name in raw):
            raise ResearchValidationError("contract field names must be strings")

        for name in _REQUIRED_FIELDS:
            if name not in raw:
                raise ResearchValidationError(f"{name} is required")

        null_hypothesis: str | None = None
        if "null_hypothesis" in raw:
            null_hypothesis_raw = raw["null_hypothesis"]
            if not isinstance(null_hypothesis_raw, str):
                raise ResearchValidationError("null_hypothesis must be a string or omitted")
            null_hypothesis = null_hypothesis_raw

        resource_budget = _optional_json_object(raw, "resource_budget")
        split_policy = _optional_json_object(raw, "split_policy")
        extensions = {name: value for name, value in raw.items() if name not in _KNOWN_FIELDS}

        return cls(
            schema_version=raw["schema_version"],
            research_id=_require_nonempty_string(raw["research_id"], "research_id"),
            question=_require_nonempty_string(raw["question"], "question"),
            hypothesis=_require_nonempty_string(raw["hypothesis"], "hypothesis"),
            null_hypothesis=null_hypothesis,
            independent_variable=_require_nonempty_string(
                raw["independent_variable"], "independent_variable"
            ),
            dependent_metrics=_require_string_list(
                raw["dependent_metrics"], "dependent_metrics", require_nonempty=True
            ),
            baselines=_require_string_list(raw["baselines"], "baselines"),
            arms=_require_string_list(raw["arms"], "arms"),
            acceptance_criteria=_require_string_list(
                raw["acceptance_criteria"], "acceptance_criteria", require_nonempty=True
            ),
            failure_criteria=_require_string_list(raw["failure_criteria"], "failure_criteria"),
            resource_budget=resource_budget,
            seed_policy=_require_json_object(raw["seed_policy"], "seed_policy"),
            split_policy=split_policy,
            repo=_require_nonempty_string(raw["repo"], "repo"),
            ref=_require_nonempty_string(raw["ref"], "ref"),
            artifact_expectations=_require_string_list(
                raw["artifact_expectations"], "artifact_expectations"
            ),
            extensions=extensions,
        )


def parse_research_json(text: str) -> ResearchContract:
    """Decode JSON text into a validated, immutable research contract."""
    if not text or not text.strip():
        raise ResearchValidationError("research contract JSON is empty")

    def _reject_non_finite(value: str) -> float:
        raise ResearchValidationError(f"research contract JSON contains non-finite number: {value}")

    try:
        raw = json.loads(text, parse_constant=_reject_non_finite)
    except json.JSONDecodeError as exc:
        raise ResearchValidationError(f"research contract is not valid JSON: {exc}") from exc
    except ResearchValidationError:
        raise
    if not isinstance(raw, dict):
        raise ResearchValidationError("research contract JSON root must be an object")
    return ResearchContract.from_dict(raw)


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchValidationError(f"{name} is required and must be a non-empty string")
    return value


def _freeze_string_array(
    value: object,
    name: str,
    *,
    require_nonempty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ResearchValidationError(f"{name} must be an array of non-empty strings")
    if require_nonempty and not value:
        raise ResearchValidationError(f"{name} must contain at least one item")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ResearchValidationError(f"{name} must be an array of non-empty strings")
    return tuple(value)


def _require_string_list(
    value: object,
    name: str,
    *,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ResearchValidationError(f"{name} must be an array of non-empty strings")
    return _freeze_string_array(value, name, require_nonempty=require_nonempty)


def _require_json_object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ResearchValidationError(f"{name} must be a JSON object")
    return _freeze_json_object(value, name)


def _optional_json_object(
    raw: dict[str, Any],
    name: str,
) -> Mapping[str, object] | None:
    if name not in raw:
        return None
    return _require_json_object(raw[name], name)


def _freeze_json_object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ResearchValidationError(f"{path} must be a JSON object")
    frozen: dict[str, object] = {}
    for name, nested in value.items():
        if not isinstance(name, str):
            raise ResearchValidationError(f"{path} field names must be strings")
        frozen[name] = _freeze_json_value(nested, f"{path}.{name}")
    return MappingProxyType(frozen)


def _freeze_json_value(value: object, path: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResearchValidationError(f"{path} must be a finite JSON number")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        return _freeze_json_object(value, path)
    raise ResearchValidationError(
        f"{path} contains unsupported JSON value of type {type(value).__name__}"
    )


def _thaw_json_object(value: Mapping[str, object]) -> dict[str, object]:
    return {name: _thaw_json_value(nested) for name, nested in value.items()}


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _thaw_json_object(value)
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value
