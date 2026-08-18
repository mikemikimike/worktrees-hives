"""Tests for Research Hive v0 roles + capability policy (GitHub #93)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from worktrees_hives.errors import PolicyError, ResearchRoleValidationError, RoleCapabilityError
from worktrees_hives.research_roles import (
    RESEARCH_ROLE_SCHEMA_VERSION,
    V0_RESEARCH_ROLES,
    ResearchCapability,
    ResearchRole,
    RoleBinding,
    assert_capability,
    assert_role_command_allowed,
    binding_metadata,
    classify_command,
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


class TestCapabilityEnforcement:
    def test_false_capability_is_denied(self) -> None:
        role = v0_role("verification_agent")
        with pytest.raises(RoleCapabilityError, match="modify_code") as exc:
            assert_capability(role, ResearchCapability.MODIFY_CODE)
        assert exc.value.code == "ROLE_CAPABILITY_DENIED"
        assert isinstance(exc.value, PolicyError)

    def test_true_capability_is_allowed(self) -> None:
        assert_capability(v0_role("verification_agent"), ResearchCapability.EXECUTE_TESTS)

    def test_verifier_cannot_git_commit(self) -> None:
        with pytest.raises(RoleCapabilityError, match="modify_code"):
            assert_role_command_allowed(v0_role("verification_agent"), ["git", "commit", "-am", "x"])

    def test_verifier_cannot_git_add_or_apply(self) -> None:
        role = v0_role("verification_agent")
        with pytest.raises(RoleCapabilityError, match="modify_code"):
            assert_role_command_allowed(role, "git add src/foo.py")
        with pytest.raises(RoleCapabilityError, match="modify_code"):
            assert_role_command_allowed(role, ["git", "apply", "change.patch"])

    def test_verifier_may_run_pytest(self) -> None:
        assert_role_command_allowed(v0_role("verification_agent"), ["pytest", "-q"])
        assert_role_command_allowed(v0_role("verification_agent"), ["python", "-m", "pytest", "-q"])

    def test_experiment_agent_may_commit(self) -> None:
        assert_role_command_allowed(v0_role("experiment_agent"), ["git", "commit", "-m", "x"])

    def test_verifier_cannot_launch_lab(self) -> None:
        with pytest.raises(RoleCapabilityError, match="launch_experiments"):
            assert_role_command_allowed(
                v0_role("verification_agent"),
                ["worktrees-hives", "lab", "run", "--hypothesis-id", "h1"],
            )

    def test_merge_still_denied_by_lab_policy(self) -> None:
        with pytest.raises(PolicyError, match="NEVER_MERGE"):
            assert_role_command_allowed(v0_role("experiment_agent"), "gh pr merge 1 --squash")

    def test_classify_commit_is_modify_code(self) -> None:
        assert ResearchCapability.MODIFY_CODE in classify_command(["git", "commit", "-m", "x"])
        assert ResearchCapability.EXECUTE_TESTS in classify_command(["pytest"])
        assert ResearchCapability.LAUNCH_EXPERIMENTS in classify_command(
            ["wh-orch", "lab", "run"]
        )
        assert classify_command(["git", "status"]) == frozenset()

    def test_classify_skips_git_globals_and_uses_basename(self) -> None:
        modify = frozenset({ResearchCapability.MODIFY_CODE})
        tests = frozenset({ResearchCapability.EXECUTE_TESTS})
        launch = frozenset({ResearchCapability.LAUNCH_EXPERIMENTS})

        assert classify_command(["git", "-C", "/tmp/repo", "commit", "-m", "x"]) == modify
        assert classify_command(["git.exe", "add", "file"]) == modify
        assert classify_command("patch -p1 < x") == modify
        assert classify_command(["patch", "file"]) == modify
        for subcommand in (
            "checkout",
            "switch",
            "reset",
            "restore",
            "rebase",
            "cherry-pick",
            "mv",
            "rm",
        ):
            assert classify_command(["git", subcommand]) == modify

        assert classify_command(["python3", "-m", "pytest"]) == tests
        assert classify_command(["python", "-m", "unittest"]) == tests
        assert classify_command(["cargo", "test"]) == tests

        assert classify_command(["lab", "run"]) == launch
        assert classify_command(["git", "log"]) == frozenset()
        assert classify_command(["git", "diff"]) == frozenset()

    def test_catalog_mapping_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            V0_RESEARCH_ROLES["verification_agent"] = v0_role("experiment_agent")  # type: ignore[index]


class TestRoleBinding:
    def test_two_models_bind_to_same_role_contract(self) -> None:
        role = v0_role("verification_agent")
        grok = RoleBinding(role=role, model_id="grok-4.6", provider="xai", agent_id="verifier-a")
        claude = RoleBinding(
            role=role, model_id="claude-opus", provider="anthropic", agent_id="verifier-b"
        )
        assert grok.role == claude.role
        assert grok.role is role
        assert grok.model_id != claude.model_id
        meta_a = binding_metadata(grok)
        meta_b = binding_metadata(claude)
        assert meta_a["role_id"] == meta_b["role_id"] == "verification_agent"
        assert meta_a["provider"] == "xai"
        assert meta_b["provider"] == "anthropic"
        assert meta_a["capabilities"]["modify_code"] is False
        assert meta_a["schema_version"] == RESEARCH_ROLE_SCHEMA_VERSION

    def test_rejects_blank_identities(self) -> None:
        role = v0_role("verification_agent")
        with pytest.raises(ResearchRoleValidationError, match="model_id"):
            RoleBinding(role=role, model_id=" ", provider="xai", agent_id="a")
        with pytest.raises(ResearchRoleValidationError, match="provider"):
            RoleBinding(role=role, model_id="grok-4.6", provider=" ", agent_id="a")
        with pytest.raises(ResearchRoleValidationError, match="agent_id"):
            RoleBinding(role=role, model_id="grok-4.6", provider="xai", agent_id=" ")


def test_package_exports_research_roles() -> None:
    import worktrees_hives as wh

    assert wh.ResearchRole is ResearchRole
    assert wh.RoleBinding is RoleBinding
    assert wh.v0_role is v0_role
    assert wh.assert_role_command_allowed is assert_role_command_allowed
