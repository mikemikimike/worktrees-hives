"""Tests for Research Hive v0 roles + capability policy (GitHub #93)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from worktrees_hives.errors import ResearchRoleValidationError
from worktrees_hives.research_roles import (
    RESEARCH_ROLE_SCHEMA_VERSION,
    V0_RESEARCH_ROLES,
    ResearchCapability,
    ResearchRole,
    parse_research_role_json,
    v0_role,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "examples" / "research-roles-v0.json"
)


def _valid_role(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "schema_version": RESEARCH_ROLE_SCHEMA_VERSION,
        "role_id": "verification_agent",
        "capabilities": {
            "read_repository": True,
            "read_results": True,
            "execute_tests": True,
            "modify_code": False,
            "launch_experiments": False,
        },
        "inputs": ["hypothesis", "experiment_manifest", "result_artifacts"],
        "outputs": ["findings.json", "verification.md"],
        "constraints": {"must_be_independent_of": ["experiment_author"]},
    }
    raw.update(overrides)
    return raw


class TestResearchRoleRoundTrip:
    def test_dict_round_trip(self) -> None:
        raw = _valid_role()
        role = ResearchRole.from_dict(raw)
        assert role.to_dict() == raw
        assert ResearchRole.from_dict(role.to_dict()) == role

    def test_json_round_trip(self) -> None:
        role = ResearchRole.from_dict(_valid_role())
        assert parse_research_role_json(role.to_json()) == role

    def test_omitted_capability_defaults_false(self) -> None:
        raw = _valid_role(capabilities={"read_repository": True})
        role = ResearchRole.from_dict(raw)
        assert role.capabilities.read_repository is True
        assert role.capabilities.modify_code is False
        assert role.capabilities.launch_experiments is False
        assert role.to_dict()["capabilities"]["modify_code"] is False

    def test_unknown_additive_fields_are_preserved(self) -> None:
        raw = _valid_role(review_protocol={"blind": True})
        role = ResearchRole.from_dict(raw)
        assert role.to_dict()["review_protocol"] == {"blind": True}

    def test_capability_vocabulary_is_exact(self) -> None:
        assert [c.value for c in ResearchCapability] == [
            "read_repository",
            "read_results",
            "execute_tests",
            "modify_code",
            "launch_experiments",
        ]

    def test_must_be_independent_of_is_derived(self) -> None:
        role = ResearchRole.from_dict(_valid_role())
        assert role.must_be_independent_of == ("experiment_author",)

    def test_allows_grants_and_denies(self) -> None:
        role = ResearchRole.from_dict(_valid_role())
        assert role.capabilities.allows(ResearchCapability.EXECUTE_TESTS) is True
        assert role.capabilities.allows(ResearchCapability.MODIFY_CODE) is False
        assert role.capabilities.allows("merge_pull_request") is False


class TestResearchRoleValidation:
    @pytest.mark.parametrize(
        "name",
        ["schema_version", "role_id", "capabilities", "inputs", "outputs", "constraints"],
    )
    def test_rejects_missing_required_field(self, name: str) -> None:
        raw = _valid_role()
        del raw[name]
        with pytest.raises(ResearchRoleValidationError, match=name):
            ResearchRole.from_dict(raw)

    def test_rejects_empty_role_id(self) -> None:
        with pytest.raises(ResearchRoleValidationError, match="role_id"):
            ResearchRole.from_dict(_valid_role(role_id="  "))

    def test_rejects_unknown_capability_key(self) -> None:
        with pytest.raises(ResearchRoleValidationError, match="unknown capability"):
            ResearchRole.from_dict(_valid_role(capabilities={"merge_pull_request": True}))

    def test_rejects_non_bool_capability(self) -> None:
        with pytest.raises(ResearchRoleValidationError, match="read_repository"):
            ResearchRole.from_dict(_valid_role(capabilities={"read_repository": "yes"}))

    def test_rejects_non_object_capabilities(self) -> None:
        with pytest.raises(ResearchRoleValidationError, match="capabilities"):
            ResearchRole.from_dict(_valid_role(capabilities=["modify_code"]))

    def test_allows_empty_inputs_and_outputs(self) -> None:
        role = ResearchRole.from_dict(_valid_role(inputs=[], outputs=[]))
        assert role.inputs == ()
        assert role.outputs == ()

    def test_rejects_blank_input(self) -> None:
        with pytest.raises(ResearchRoleValidationError, match="inputs"):
            ResearchRole.from_dict(_valid_role(inputs=["hypothesis", ""]))

    def test_rejects_non_list_must_be_independent_of(self) -> None:
        with pytest.raises(ResearchRoleValidationError, match="must_be_independent_of"):
            ResearchRole.from_dict(
                _valid_role(constraints={"must_be_independent_of": "experiment_author"})
            )

    def test_constraints_may_be_empty(self) -> None:
        role = ResearchRole.from_dict(_valid_role(constraints={}))
        assert role.must_be_independent_of == ()

    def test_rejects_invalid_schema_version(self) -> None:
        with pytest.raises(ResearchRoleValidationError, match="schema_version"):
            ResearchRole.from_dict(_valid_role(schema_version=2))

    def test_rejects_malformed_json(self) -> None:
        with pytest.raises(ResearchRoleValidationError):
            parse_research_role_json("{not json")


class TestResearchRoleImmutability:
    def test_input_mutation_cannot_change_role(self) -> None:
        raw = _valid_role()
        role = ResearchRole.from_dict(raw)
        raw["inputs"].append("late")
        raw["capabilities"]["modify_code"] = True
        assert "late" not in role.inputs
        assert role.capabilities.modify_code is False

    def test_frozen(self) -> None:
        role = ResearchRole.from_dict(_valid_role())
        with pytest.raises(FrozenInstanceError):
            role.role_id = "other"  # type: ignore[misc]


class TestV0Catalog:
    def test_four_role_ids(self) -> None:
        assert tuple(V0_RESEARCH_ROLES) == (
            "research_coordinator",
            "experiment_agent",
            "verification_agent",
            "artifact_agent",
        )

    def test_verification_agent_cannot_modify_or_launch(self) -> None:
        role = v0_role("verification_agent")
        assert role.capabilities.modify_code is False
        assert role.capabilities.launch_experiments is False
        assert role.capabilities.execute_tests is True
        assert role.must_be_independent_of == ("experiment_author",)

    def test_experiment_agent_may_modify_and_launch(self) -> None:
        role = v0_role("experiment_agent")
        assert role.capabilities.modify_code is True
        assert role.capabilities.launch_experiments is True

    def test_coordinator_and_artifact_are_non_mutating(self) -> None:
        for role_id in ("research_coordinator", "artifact_agent"):
            role = v0_role(role_id)
            assert role.capabilities.modify_code is False
            assert role.capabilities.execute_tests is False
            assert role.capabilities.launch_experiments is False
            assert role.capabilities.read_repository is True
            assert role.capabilities.read_results is True

    def test_unknown_catalog_id_raises(self) -> None:
        with pytest.raises(ResearchRoleValidationError, match="unknown v0 role"):
            v0_role("literature_agent")

    def test_fixture_round_trip(self) -> None:
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        roles = [ResearchRole.from_dict(item) for item in payload["roles"]]
        assert [r.role_id for r in roles] == list(V0_RESEARCH_ROLES)
        for role in roles:
            assert role == v0_role(role.role_id)
