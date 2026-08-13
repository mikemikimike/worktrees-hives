"""Tests for the Research Hive v0 experiment contract (GitHub #92)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from worktrees_hives.contract import Response
from worktrees_hives.errors import ResearchValidationError
from worktrees_hives.research import (
    RESEARCH_CONTRACT_SCHEMA_VERSION,
    ResearchContract,
    ResearchOutcome,
    parse_research_json,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "examples" / "research-contract-cloud-agent.json"
)


def _valid_contract(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "schema_version": RESEARCH_CONTRACT_SCHEMA_VERSION,
        "research_id": "structured-contract-eval-v1",
        "question": "Does structured context improve cloud coding-agent evidence?",
        "hypothesis": "Structured context improves first-pass valid evidence by 10 points.",
        "null_hypothesis": "Structured context does not improve valid evidence.",
        "independent_variable": "task_context_format",
        "dependent_metrics": ["first_pass_valid_evidence_rate", "median_input_tokens"],
        "baselines": ["issue_text_only"],
        "arms": ["issue_text_plus_contract"],
        "acceptance_criteria": [
            "first_pass_valid_evidence_rate_delta >= 0.10",
            "median_input_tokens_relative_delta <= 0.15",
        ],
        "failure_criteria": ["paired_task_completion_rate < 0.90"],
        "resource_budget": {
            "max_agent_runs": 40,
            "limits": {"wall_time_minutes": 240},
        },
        "seed_policy": {
            "strategy": "fixed_per_task",
            "seeds": [104729, 130363],
        },
        "split_policy": {
            "unit": "repository_task",
            "evaluation_task_ids": ["schema-01", "contract-02"],
        },
        "repo": "acme/agent-eval-harness",
        "ref": "8f4c2d91a6b753e0c18d2a946f0b73e5a1c29d84",
        "artifact_expectations": ["metrics/paired-runs.json", "reports/verification.md"],
    }
    raw.update(overrides)
    return raw


class TestResearchContractRoundTrip:
    def test_dict_round_trip(self) -> None:
        raw = _valid_contract()
        contract = ResearchContract.from_dict(raw)

        assert contract.to_dict() == raw
        assert ResearchContract.from_dict(contract.to_dict()) == contract

    def test_json_round_trip(self) -> None:
        contract = ResearchContract.from_dict(_valid_contract())

        assert parse_research_json(contract.to_json()) == contract

    def test_realistic_cloud_agent_fixture(self) -> None:
        contract = parse_research_json(FIXTURE_PATH.read_text(encoding="utf-8"))

        assert contract.research_id == "structured-contract-first-pass-evidence-v1"
        assert contract.baselines == ("issue_text_only",)
        assert contract.arms == ("issue_text_plus_frozen_research_contract",)
        assert contract.resource_budget is not None
        assert contract.resource_budget["max_agent_runs"] == 40
        assert contract.seed_policy["reuse_across_arms"] is True
        assert contract.repo == "acme/agent-eval-harness"

    def test_unknown_additive_fields_are_preserved(self) -> None:
        future_field = {
            "review_protocol": {
                "models": ["cloud-model-a", "cloud-model-b"],
                "blind": True,
            }
        }
        raw = _valid_contract(future_evidence=future_field)

        contract = ResearchContract.from_dict(raw)
        serialized = contract.to_dict()

        assert serialized["future_evidence"] == future_field
        assert ResearchContract.from_dict(serialized).to_dict() == raw

    def test_transport_and_domain_versions_are_independent(self) -> None:
        envelope = Response.from_dict(
            {
                "ok": True,
                "schema_version": 1,
                "command": "lab.example",
                "data": {"research_contract": _valid_contract()},
                "error": None,
            }
        )

        contract = ResearchContract.from_dict(envelope.data["research_contract"])

        assert envelope.schema_version == 1
        assert contract.schema_version == RESEARCH_CONTRACT_SCHEMA_VERSION


class TestOutcomeVocabulary:
    def test_exact_four_values(self) -> None:
        assert [outcome.value for outcome in ResearchOutcome] == [
            "supported",
            "not_supported",
            "inconclusive",
            "invalid",
        ]

    def test_outcome_is_not_part_of_pre_execution_contract(self) -> None:
        assert "outcome" not in {field.name for field in fields(ResearchContract)}
        assert "outcome" not in ResearchContract.from_dict(_valid_contract()).to_dict()


class TestRequiredness:
    @pytest.mark.parametrize(
        "name",
        [
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
        ],
    )
    def test_rejects_missing_required_field(self, name: str) -> None:
        raw = _valid_contract()
        del raw[name]

        with pytest.raises(ResearchValidationError, match=rf"{name} is required"):
            ResearchContract.from_dict(raw)

    @pytest.mark.parametrize(
        "name",
        ["research_id", "question", "hypothesis", "independent_variable", "repo", "ref"],
    )
    @pytest.mark.parametrize("value", ["", "   "])
    def test_rejects_empty_required_string(self, name: str, value: str) -> None:
        with pytest.raises(ResearchValidationError, match=name):
            ResearchContract.from_dict(_valid_contract(**{name: value}))

    @pytest.mark.parametrize("name", ["dependent_metrics", "acceptance_criteria"])
    def test_rejects_empty_nonempty_array(self, name: str) -> None:
        with pytest.raises(ResearchValidationError, match="at least one"):
            ResearchContract.from_dict(_valid_contract(**{name: []}))

    @pytest.mark.parametrize(
        "name",
        ["baselines", "arms", "failure_criteria", "artifact_expectations"],
    )
    def test_allows_empty_presence_only_array(self, name: str) -> None:
        contract = ResearchContract.from_dict(_valid_contract(**{name: []}))

        assert getattr(contract, name) == ()

    def test_optional_fields_may_be_omitted(self) -> None:
        raw = _valid_contract()
        for name in ("null_hypothesis", "resource_budget", "split_policy"):
            del raw[name]

        contract = ResearchContract.from_dict(raw)

        assert contract.null_hypothesis is None
        assert contract.resource_budget is None
        assert contract.split_policy is None
        assert contract.to_dict() == raw

    def test_optional_fields_must_have_their_declared_type_when_present(self) -> None:
        with pytest.raises(ResearchValidationError, match="null_hypothesis"):
            ResearchContract.from_dict(_valid_contract(null_hypothesis=None))

    def test_seed_policy_may_be_empty_but_must_be_present(self) -> None:
        contract = ResearchContract.from_dict(_valid_contract(seed_policy={}))

        assert dict(contract.seed_policy) == {}


class TestFailClosedValidation:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("dependent_metrics", ["valid", ""]),
            ("baselines", [42]),
            ("arms", "treatment"),
            ("arms", ("treatment",)),
            ("acceptance_criteria", [None]),
            ("failure_criteria", ["   "]),
            ("artifact_expectations", {"path": "report.md"}),
        ],
    )
    def test_rejects_invalid_collection_members(self, name: str, value: object) -> None:
        with pytest.raises(ResearchValidationError, match=name):
            ResearchContract.from_dict(_valid_contract(**{name: value}))

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("seed_policy", "random"),
            ("seed_policy", MappingProxyType({})),
            ("resource_budget", []),
            ("split_policy", "by-session"),
        ],
    )
    def test_rejects_non_object_policy_or_budget(self, name: str, value: object) -> None:
        with pytest.raises(ResearchValidationError, match="JSON object"):
            ResearchContract.from_dict(_valid_contract(**{name: value}))

    @pytest.mark.parametrize("version", [True, "1", 2])
    def test_rejects_invalid_schema_version(self, version: object) -> None:
        with pytest.raises(ResearchValidationError, match="schema_version"):
            ResearchContract.from_dict(_valid_contract(schema_version=version))

    def test_reports_unsupported_schema_before_v1_field_errors(self) -> None:
        raw = _valid_contract(schema_version=2)
        del raw["research_id"]

        with pytest.raises(ResearchValidationError, match="unsupported schema_version 2"):
            ResearchContract.from_dict(raw)

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_rejects_non_finite_json_number(self, constant: str) -> None:
        raw = _valid_contract(resource_budget={"limit": "__NON_FINITE__"})
        text = json.dumps(raw).replace('"__NON_FINITE__"', constant, 1)

        with pytest.raises(ResearchValidationError, match="non-finite number"):
            parse_research_json(text)

    def test_rejects_overflow_number_literal(self) -> None:
        raw = _valid_contract(resource_budget={"limit": "__OVERFLOW__"})
        text = json.dumps(raw).replace('"__OVERFLOW__"', "1e400", 1)

        with pytest.raises(ResearchValidationError, match="finite JSON number"):
            parse_research_json(text)

    def test_rejects_nested_non_finite_number_from_dict(self) -> None:
        with pytest.raises(ResearchValidationError, match="finite JSON number"):
            ResearchContract.from_dict(
                _valid_contract(resource_budget={"nested": {"limit": float("nan")}})
            )

    def test_rejects_non_string_field_name(self) -> None:
        raw = _valid_contract()
        raw[1] = "numeric key"  # type: ignore[index]

        with pytest.raises(ResearchValidationError, match="field names must be strings"):
            ResearchContract.from_dict(raw)

    def test_rejects_non_json_additive_value(self) -> None:
        with pytest.raises(ResearchValidationError, match="unsupported JSON value"):
            ResearchContract.from_dict(_valid_contract(future_value=object()))

    def test_rejects_extensions_that_shadow_known_fields(self) -> None:
        contract = ResearchContract.from_dict(_valid_contract())

        with pytest.raises(ResearchValidationError, match="cannot shadow known fields"):
            replace(contract, extensions={"hypothesis": "shadowed"})

    @pytest.mark.parametrize("text", ["", "   ", "[]", "{not json"])
    def test_rejects_malformed_json_document(self, text: str) -> None:
        with pytest.raises(ResearchValidationError):
            parse_research_json(text)


class TestImmutability:
    def test_input_mutation_cannot_change_contract(self) -> None:
        raw = _valid_contract(future_field={"nested": ["original"]})
        contract = ResearchContract.from_dict(raw)

        raw["acceptance_criteria"].append("late criterion")
        raw["seed_policy"]["seeds"].append(999983)
        raw["future_field"]["nested"].append("late extension")

        assert "late criterion" not in contract.acceptance_criteria
        assert contract.seed_policy["seeds"] == (104729, 130363)
        assert contract.to_dict()["future_field"] == {"nested": ["original"]}

    def test_attributes_and_nested_collections_are_read_only(self) -> None:
        contract = ResearchContract.from_dict(_valid_contract())

        with pytest.raises(FrozenInstanceError):
            contract.hypothesis = "changed"  # type: ignore[misc]
        with pytest.raises(TypeError):
            contract.seed_policy["strategy"] = "random"  # type: ignore[index]
        with pytest.raises(AttributeError):
            contract.seed_policy["seeds"].append(999983)  # type: ignore[union-attr]

    def test_serialized_output_is_a_defensive_copy(self) -> None:
        contract = ResearchContract.from_dict(_valid_contract())
        serialized = contract.to_dict()

        serialized["acceptance_criteria"].append("late criterion")
        serialized["resource_budget"]["limits"]["wall_time_minutes"] = 999

        again = contract.to_dict()
        assert "late criterion" not in again["acceptance_criteria"]
        assert again["resource_budget"]["limits"]["wall_time_minutes"] == 240
