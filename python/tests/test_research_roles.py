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

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "docs" / "examples" / "research-roles-v0.json"


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

    def test_normalizes_role_id_whitespace(self) -> None:
        role = ResearchRole.from_dict(_valid_role(role_id=" verification_agent "))
        assert role.role_id == "verification_agent"

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

    def test_unknown_capability_is_a_structured_policy_denial(self) -> None:
        with pytest.raises(RoleCapabilityError, match="merge_pull_request") as exc:
            assert_capability(v0_role("verification_agent"), "merge_pull_request")
        assert exc.value.code == "ROLE_CAPABILITY_DENIED"

    def test_verifier_cannot_git_commit(self) -> None:
        with pytest.raises(RoleCapabilityError, match="modify_code"):
            assert_role_command_allowed(
                v0_role("verification_agent"), ["git", "commit", "-am", "x"]
            )

    def test_verifier_cannot_git_add_or_apply(self) -> None:
        role = v0_role("verification_agent")
        with pytest.raises(RoleCapabilityError, match="modify_code"):
            assert_role_command_allowed(role, "git add src/foo.py")
        with pytest.raises(RoleCapabilityError, match="modify_code"):
            assert_role_command_allowed(role, ["git", "apply", "change.patch"])

    def test_verifier_cannot_use_command_launching_git_options(self) -> None:
        for command in (
            ["git", "grep", "--open-files-in-pager=sh -c 'touch victim'", "pattern"],
            ["git", "grep", "--open-files-in-p=sh -c 'touch victim'", "pattern"],
            ["git", "cat-file", "--filters", "HEAD:victim"],
            ["git", "cat-file", "--textconv", "HEAD:victim"],
        ):
            with pytest.raises(RoleCapabilityError, match="modify_code"):
                assert_role_command_allowed(v0_role("verification_agent"), command)

    @pytest.mark.parametrize(
        "command",
        [
            ["env", "git", "commit", "-m", "x"],
            ["env", "-S", "git add victim"],
            ["env", "-S", r"git\_add victim"],
            ["env", "-Sgit add victim"],
            ["env", "-vS", "git add victim"],
            ["env", "--split-string=git add victim"],
            ["env", "FOO=git", "env", "-S", "${FOO} add victim"],
            ["env", "-", "git", "add", "victim"],
            ["env", "-a", "spoof", "git", "add", "victim"],
            ["env", "-P", "/opt/tools", "git", "add", "victim"],
            ["timeout", "10", "git", "stash", "pop"],
            ["timeout", "--sig", "TERM", "10", "git", "add", "victim"],
            ["timeout", "-vk", "2", "3", "git", "add", "victim"],
            ["nice", "git", "clean", "-fd"],
            ["sh", "-c", "git commit -m x"],
            ["ash", "-c", "git commit -m x"],
            ["csh", "-c", "git commit -m x"],
            ["dash", "-c", "git commit -m x"],
            ["tcsh", "-c", "git commit -m x"],
        ],
    )
    def test_verifier_cannot_bypass_modify_gate_with_wrappers(self, command: list[str]) -> None:
        with pytest.raises(RoleCapabilityError, match="modify_code"):
            assert_role_command_allowed(v0_role("verification_agent"), command)

    def test_verifier_may_run_pytest(self) -> None:
        assert_role_command_allowed(v0_role("verification_agent"), ["pytest", "-q"])
        assert_role_command_allowed(v0_role("verification_agent"), ["python", "-m", "pytest", "-q"])

    def test_shell_wrappers_retain_nested_capability_requirements(self) -> None:
        modifier_without_launch = ResearchRole.from_dict(
            _valid_role(
                role_id="modifier_without_launch",
                capabilities={
                    "read_repository": True,
                    "read_results": True,
                    "execute_tests": True,
                    "modify_code": True,
                    "launch_experiments": False,
                },
            )
        )
        for command in (
            ["sh", "-c", "worktrees-hives lab run --hypothesis-id h1"],
            ["python", "-c", "print('arbitrary code')"],
            ["python", "script.py"],
            ["python", "-m", "http.server"],
            ["python", "-i", "-m", "pytest"],
            ["git", "grep", "--open-files-in-pager=sh", "pattern"],
            ["git", "cat-file", "--filters", "HEAD:victim"],
            ["git", "-c", "core.pager=sh", "--paginate", "log"],
            ["git", "--paginate", "log"],
            ["git", "diff", "--ext-diff"],
            ["git", "show", "--textconv", "HEAD:victim"],
            ["git", "diff"],
            ["git", "log", "-p"],
            ["git", "show", "HEAD:victim"],
            ["git", "grep", "--textconv", "pattern"],
            ["git", "blame", "--textconv", "HEAD:victim"],
            ["git", "help", "status"],
            ["git", "remote", "show", "origin"],
            ["git", "log", "--show-signature"],
            ["git", "show", "--format=%G?", "HEAD"],
            ["git", "status"],
            ["git", "log", "-up", "-1"],
            ["git", "whatchanged", "-p", "-1"],
            ["git", "verify-commit", "HEAD"],
            ["git", "verify-tag", "v1"],
            ["git", "tag", "-v", "v1"],
            ["git", "ls-remote", "evil"],
            ["git", "credential", "fill"],
            ["git", "difftool", "HEAD~1", "HEAD"],
            [
                "git",
                "-P",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "log.showSignature=false",
                "whatchanged",
                "--pretty=format:%H",
                "-p",
                "--no-textconv",
                "--ext-diff",
                "-1",
            ],
            ["git", "-P", "status"],
            [
                "git",
                "-P",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "log.showSignature=false",
                "log",
                "--pretty=verify",
                "-1",
            ],
            ["git", "archive", "--format=evil", "HEAD"],
            ["env", "-P", "/tmp/attacker", "git", "status"],
            ["env", "X=1", "worktrees-hives", "lab", "run"],
        ):
            with pytest.raises(RoleCapabilityError, match="launch_experiments"):
                assert_role_command_allowed(modifier_without_launch, command)

        modifier_without_tests = ResearchRole.from_dict(
            _valid_role(
                role_id="modifier_without_tests",
                capabilities={
                    "read_repository": True,
                    "read_results": True,
                    "execute_tests": False,
                    "modify_code": True,
                    "launch_experiments": True,
                },
            )
        )
        for command in (
            ["sh", "-c", "pytest -q"],
            ["python", "-"],
            ["git", "grep", "-Osh", "pattern"],
            ["git", "cat-file", "--textconv", "HEAD:victim"],
            ["git", "log", "--ext-diff", "-p"],
            ["env", "CI=1", "pytest", "-q"],
            ["worktrees-hives", "lab", "run", "--command", "pytest -q"],
        ):
            with pytest.raises(RoleCapabilityError, match="execute_tests"):
                assert_role_command_allowed(modifier_without_tests, command)

        launcher_without_modify = ResearchRole.from_dict(
            _valid_role(
                role_id="launcher_without_modify",
                capabilities={
                    "read_repository": True,
                    "read_results": True,
                    "execute_tests": True,
                    "modify_code": False,
                    "launch_experiments": True,
                },
            )
        )
        for command in (
            ["lab", "run", "--command", "git add victim"],
            [
                "python",
                "-m",
                "worktrees_hives.cli",
                "lab",
                "run",
                "--command=git add victim",
            ],
        ):
            with pytest.raises(RoleCapabilityError, match="modify_code"):
                assert_role_command_allowed(launcher_without_modify, command)

    def test_versioned_python_names_cannot_bypass_capability_checks(self) -> None:
        for executable in (
            "python3.14",
            "python3.14t",
            "pythonw.exe",
            "pythonw3.14.exe",
            "pythonw3.14t.exe",
            "pypy3",
            "pyw.exe",
        ):
            with pytest.raises(RoleCapabilityError, match="launch_experiments"):
                assert_role_command_allowed(
                    v0_role("verification_agent"),
                    [executable, "-m", "worktrees_hives.cli", "lab", "run"],
                )
        for executable in (
            "python3.14.exe",
            "pythonw",
            "pythonw3.14",
            "pypy3.11.exe",
            "pyw",
        ):
            with pytest.raises(RoleCapabilityError, match="execute_tests"):
                assert_role_command_allowed(v0_role("artifact_agent"), [executable, "-m", "pytest"])

    @pytest.mark.parametrize("module", ["pytest.__main__", "unittest.__main__"])
    def test_test_package_main_modules_require_execute_tests(self, module: str) -> None:
        with pytest.raises(RoleCapabilityError, match="execute_tests"):
            assert_role_command_allowed(v0_role("artifact_agent"), ["python", "-m", module])

    def test_experiment_agent_may_commit(self) -> None:
        assert_role_command_allowed(v0_role("experiment_agent"), ["git", "commit", "-m", "x"])

    def test_verifier_cannot_launch_lab(self) -> None:
        role = v0_role("verification_agent")
        with pytest.raises(RoleCapabilityError, match="launch_experiments"):
            assert_role_command_allowed(
                role,
                ["worktrees-hives", "lab", "run", "--hypothesis-id", "h1"],
            )
        with pytest.raises(RoleCapabilityError, match="launch_experiments"):
            assert_role_command_allowed(
                role,
                ["worktrees-hives", "--s", "/tmp/x", "lab", "run"],
            )
        with pytest.raises(RoleCapabilityError, match="launch_experiments"):
            assert_role_command_allowed(
                role,
                ["python", "-m", "worktrees_hives.cli", "lab", "run"],
            )

    def test_merge_still_denied_by_lab_policy(self) -> None:
        with pytest.raises(PolicyError, match="NEVER_MERGE"):
            assert_role_command_allowed(v0_role("experiment_agent"), "gh pr merge 1 --squash")

    def test_classify_commit_is_modify_code(self) -> None:
        assert ResearchCapability.MODIFY_CODE in classify_command(["git", "commit", "-m", "x"])
        assert ResearchCapability.EXECUTE_TESTS in classify_command(["pytest"])
        assert ResearchCapability.LAUNCH_EXPERIMENTS in classify_command(["wh-orch", "lab", "run"])
        assert (
            classify_command(
                [
                    "git",
                    "--no-pager",
                    "-c",
                    "core.fsmonitor=false",
                    "status",
                ]
            )
            == frozenset()
        )

    def test_classify_skips_git_globals_and_uses_basename(self) -> None:
        modify = frozenset({ResearchCapability.MODIFY_CODE})
        tests = frozenset({ResearchCapability.EXECUTE_TESTS})
        launch = frozenset({ResearchCapability.LAUNCH_EXPERIMENTS})
        shell = frozenset(
            {
                ResearchCapability.EXECUTE_TESTS,
                ResearchCapability.MODIFY_CODE,
                ResearchCapability.LAUNCH_EXPERIMENTS,
            }
        )
        safe_git = [
            "git",
            "-P",
            "-c",
            "core.fsmonitor=false",
        ]
        safe_history_git = [
            *safe_git,
            "-c",
            "log.showSignature=false",
        ]

        assert classify_command(["git", "-C", "/tmp/repo", "commit", "-m", "x"]) == modify
        assert classify_command(["git.exe", "add", "file"]) == modify
        assert classify_command("patch -p1 < x") == modify
        assert classify_command(["patch", "file"]) == modify
        for subcommand in (
            "am",
            "checkout",
            "clean",
            "merge",
            "pull",
            "switch",
            "reset",
            "restore",
            "rebase",
            "cherry-pick",
            "revert",
            "stash",
            "mv",
            "rm",
        ):
            assert classify_command(["git", subcommand]) == modify

        assert classify_command(["env", "git", "commit", "-m", "x"]) == modify
        assert (
            classify_command(
                [
                    "env",
                    "-u",
                    "CI",
                    "timeout",
                    "--signal",
                    "TERM",
                    "5",
                    "nice",
                    "-n",
                    "2",
                    "git",
                    "commit",
                    "-m",
                    "x",
                ]
            )
            == modify
        )
        assert classify_command(["sh", "-c", "git commit -m x"]) == shell
        assert classify_command(["ash", "-c", "git commit -m x"]) == shell
        assert classify_command(["csh", "-c", "git commit -m x"]) == shell
        assert classify_command(["dash", "-c", "git commit -m x"]) == shell
        assert classify_command(["tcsh", "-c", "git commit -m x"]) == shell
        assert classify_command(["git", "-C"]) == frozenset()
        assert classify_command([*safe_git, "branch", "--show-current"]) == frozenset()
        assert classify_command([*safe_git, "remote", "-v"]) == frozenset()
        assert classify_command([*safe_git, "config", "--get", "user.name"]) == frozenset()
        assert classify_command(["git", "branch", "new-branch"]) == modify
        assert classify_command(["git", "remote", "add", "origin", "url"]) == modify
        assert classify_command(["git", "remote", "-v", "add", "origin", "url"]) == modify
        assert classify_command(["git", "config", "user.name", "value"]) == modify
        assert classify_command(["git", "diff", "--output=victim.py"]) == shell
        assert classify_command(["git", "show", "--output", "victim.py"]) == shell
        assert (
            classify_command(
                ["git", "grep", "--open-files-in-pager=sh -c 'touch victim'", "pattern"]
            )
            == shell
        )
        assert (
            classify_command(["git", "grep", "--open-files-in-p=sh -c 'touch victim'", "pattern"])
            == shell
        )
        assert classify_command(["git", "grep", "-Osh", "pattern"]) == shell
        assert classify_command(["git", "cat-file", "--filters", "HEAD:victim"]) == shell
        assert classify_command(["git", "cat-file", "--textconv", "HEAD:victim"]) == shell
        assert classify_command(["git", "diff", "--ext-diff"]) == shell
        assert classify_command(["git", "show", "--textconv", "HEAD:victim"]) == shell
        assert classify_command(["git", "log", "--ext", "-p"]) == shell
        assert classify_command(["git", "diff"]) == shell
        assert classify_command(["git", "log", "-p"]) == shell
        assert classify_command(["git", "show", "HEAD:victim"]) == shell
        assert classify_command(["git", "grep", "--textc", "pattern"]) == shell
        assert classify_command(["git", "blame", "--textconv", "HEAD:victim"]) == shell
        assert classify_command(["git", "help", "status"]) == shell
        assert classify_command(["git", "remote", "show", "origin"]) == shell
        assert classify_command([*safe_git, "remote", "show", "-n", "origin"]) == frozenset()
        assert classify_command(["git", "log", "--show-sign", "HEAD"]) == shell
        assert classify_command(["git", "show", "--format=%G?", "HEAD"]) == shell
        assert classify_command(["git", "status"]) == shell
        assert classify_command(["git", "log"]) == shell
        assert classify_command(["git", "--paginate", "status"]) == shell
        assert classify_command(["git", "--no-pager", "--paginate", "status"]) == shell
        assert (
            classify_command(
                [
                    "git",
                    "--paginate",
                    "--no-pager",
                    "-c",
                    "core.fsmonitor=false",
                    "status",
                ]
            )
            == frozenset()
        )
        assert classify_command(["git", "-P", "status"]) == shell
        assert (
            classify_command(["git", "-P", "-c", "core.fsmonitor=false", "status"]) == frozenset()
        )
        assert classify_command(["git", "-P", "-c", "log.showSignature=false", "status"]) == shell
        assert (
            classify_command(
                [
                    "git",
                    "-P",
                    "-ccore.fsmonitor=false",
                    "status",
                ]
            )
            == frozenset()
        )
        assert classify_command(["git", "-P", "-c", "core.fsmonitor=true", "status"]) == shell
        assert (
            classify_command([*safe_git, "diff", "--no-ext-diff", "--no-textconv"]) == frozenset()
        )
        assert (
            classify_command(
                [*safe_history_git, "show", "--pretty=format:%H", "--no-textconv", "HEAD"]
            )
            == frozenset()
        )
        assert (
            classify_command(
                [*safe_history_git, "log", "--pretty=format:%H", "-p", "--no-textconv"]
            )
            == frozenset()
        )
        assert classify_command([*safe_git, "cat-file", "-p", "HEAD:victim"]) == frozenset()
        assert classify_command(["env", "-P", "/tmp/attacker", "git", "status"]) == shell
        assert classify_command(["git", "log", "-up", "-1"]) == shell
        assert classify_command(["git", "whatchanged", "-p", "-1"]) == shell
        assert (
            classify_command(
                [*safe_history_git, "log", "--pretty=format:%H", "-up", "--no-textconv"]
            )
            == frozenset()
        )
        assert (
            classify_command(
                [
                    *safe_history_git,
                    "whatchanged",
                    "--pretty=format:%H",
                    "-p",
                    "--no-textconv",
                ]
            )
            == frozenset()
        )
        assert (
            classify_command([*safe_history_git, "log", "--pretty=format:%H", "-up", "-s"])
            == frozenset()
        )
        assert (
            classify_command([*safe_history_git, "log", "--pretty=format:%H", "-s", "-up"]) == shell
        )
        assert (
            classify_command(
                [
                    *safe_history_git,
                    "whatchanged",
                    "--pretty=format:%H",
                    "-p",
                    "--no-textconv",
                    "--ext-diff",
                    "-1",
                ]
            )
            == shell
        )
        assert classify_command(["git", "verify-commit", "HEAD"]) == shell
        assert classify_command(["git", "verify-tag", "v1"]) == shell
        assert classify_command(["git", "tag", "-v", "v1"]) == shell
        assert classify_command(["git", "tag", "-lv", "v1"]) == shell
        assert classify_command(["git", "tag", "--verify", "v1"]) == shell
        assert classify_command(["git", "ls-remote", "evil"]) == shell
        assert classify_command([*safe_git, "ls-remote", "--get-url", "evil"]) == frozenset()
        assert classify_command(["git", "credential", "fill"]) == shell
        assert classify_command(["git", "difftool", "HEAD~1", "HEAD"]) == shell
        assert classify_command(["git", "archive", "--format=evil", "HEAD"]) == shell
        assert classify_command([*safe_git, "log", "--pretty=medium", "-1"]) == shell
        assert classify_command([*safe_history_git, "log"]) == shell
        assert classify_command([*safe_history_git, "log", "--pretty=verify", "-1"]) == shell
        assert classify_command([*safe_history_git, "log", "--format=verify", "-1"]) == shell
        assert classify_command([*safe_history_git, "rev-list", "--pretty=verify", "HEAD"]) == shell
        assert classify_command([*safe_history_git, "shortlog", "--format=verify", "HEAD"]) == shell
        assert classify_command([*safe_history_git, "log", "--pretty=medium", "-1"]) == frozenset()
        assert (
            classify_command([*safe_history_git, "log", "--pretty=format:%H", "-1"]) == frozenset()
        )
        assert (
            classify_command([*safe_history_git, "log", "--format=tformat:%H", "-1"]) == frozenset()
        )
        assert classify_command([*safe_history_git, "log", "--oneline", "-1"]) == frozenset()
        assert classify_command([*safe_history_git, "log", "--pretty=format:%G?", "-1"]) == shell
        assert classify_command(["git", "diff", "--", "--output=victim.py"]) == shell
        assert (
            classify_command(["git", "-c", "diff.external=sh -c 'touch victim'", "diff"]) == shell
        )
        assert classify_command(["git", "--pag", "log"]) == shell
        assert classify_command(["echo", "git", "commit"]) == frozenset()
        assert classify_command(["echo", "worktrees-hives", "lab"]) == frozenset()
        assert classify_command(["node", "-e", "process.exit(0)"]) == frozenset()
        assert classify_command(["cargo", "run"]) == frozenset()

        assert classify_command(["py.test", "-q"]) == tests
        assert classify_command(["python3", "-m", "pytest"]) == tests
        assert classify_command(["python3.14", "-m", "pytest"]) == tests
        assert classify_command(["python3.14.exe", "-m", "pytest"]) == tests
        assert classify_command(["pythonw", "-m", "pytest"]) == tests
        assert classify_command(["pythonw.exe", "-m", "pytest"]) == tests
        assert classify_command(["pythonw3.14", "-m", "pytest"]) == tests
        assert classify_command(["pythonw3.14.exe", "-m", "pytest"]) == tests
        assert classify_command(["python3.14t", "-m", "pytest"]) == tests
        assert classify_command(["pythonw3.14t.exe", "-m", "pytest"]) == tests
        assert classify_command(["pypy3", "-m", "pytest"]) == tests
        assert classify_command(["pypy3.11.exe", "-m", "pytest"]) == tests
        assert classify_command(["pyw", "-m", "pytest"]) == tests
        assert classify_command(["pyw.exe", "-m", "pytest"]) == tests
        assert classify_command(["python3", "-mpytest"]) == tests
        assert classify_command(["python3", "-Bm", "pytest"]) == tests
        assert classify_command(["python3", "-Bmpytest"]) == tests
        assert classify_command(["python", "-m", "unittest"]) == tests
        assert classify_command(["python", "-m", "pytest.__main__"]) == tests
        assert classify_command(["python", "-m", "unittest.__main__"]) == tests
        assert classify_command(["python", "-i", "-m", "pytest"]) == shell
        assert classify_command(["python", "-Bimpytest"]) == shell
        assert classify_command(["python", "helper.py", "-m", "pytest"]) == shell
        assert classify_command(["python", "-c", "pass", "-m", "pytest"]) == shell
        assert classify_command(["python", "-m", "http.server"]) == shell
        assert classify_command(["cargo", "test"]) == tests
        assert classify_command(["cargo", "t"]) == tests
        assert classify_command(["cargo", "+nightly", "test"]) == tests
        assert classify_command(["cargo", "build", "--package", "test"]) == frozenset()

        assert classify_command(["lab", "run"]) == launch
        assert classify_command(["lab", "run", "--command", "git add victim"]) == shell
        assert classify_command(["python", "-m", "worktrees_hives.cli", "lab", "run"]) == launch
        assert (
            classify_command(
                [
                    "python",
                    "-m",
                    "worktrees_hives.cli",
                    "lab",
                    "run",
                    "--command=pytest -q",
                ]
            )
            == shell
        )
        assert classify_command(["python", "-mworktrees_hives.cli", "lab", "run"]) == launch
        assert classify_command(["python", "-Bm", "worktrees_hives.cli", "lab", "run"]) == launch
        assert classify_command(["py", "-m", "worktrees_hives.cli", "lab", "run"]) == launch
        assert classify_command(["py.exe", "-m", "worktrees_hives.cli", "lab", "run"]) == launch
        assert classify_command(["worktrees-hives", "--json", "lab", "run"]) == launch
        assert classify_command(["wh-orch", "--state", "/tmp/lab", "lab", "run"]) == launch
        assert classify_command(["worktrees-hives", "--s", "/tmp/x", "lab", "run"]) == launch
        assert classify_command(["wh-orch", "--sta", "/tmp/x", "lab", "run"]) == launch
        assert classify_command(["wh-orch", "lab", "run", "--command", "git add victim"]) == shell
        assert classify_command(["worktrees-hives", "plan", "--repo", "org/lab"]) == frozenset()
        assert (
            classify_command(
                ["wh-orch", "watchlist", "add", "j1", "acme", "example", "feature/lab"]
            )
            == frozenset()
        )
        assert classify_command([*safe_history_git, "log", "--pretty=format:%H"]) == frozenset()

    def test_env_assignments_require_full_capabilities(self) -> None:
        shell = frozenset(
            {
                ResearchCapability.EXECUTE_TESTS,
                ResearchCapability.MODIFY_CODE,
                ResearchCapability.LAUNCH_EXPERIMENTS,
            }
        )
        role = v0_role("artifact_agent")

        for assignment in (
            "GIT_EXTERNAL_DIFF=sh -c 'touch victim' -",
            "PATH=/tmp/attacker",
            "PAGER=sh -c 'touch victim'",
            "CI=1",
        ):
            command = ["env", assignment, "git", "diff"]
            assert classify_command(command) == shell
            with pytest.raises(RoleCapabilityError, match="execute_tests"):
                assert_role_command_allowed(role, command)

            separated_command = ["env", "--", assignment, "git", "diff"]
            assert classify_command(separated_command) == shell
            with pytest.raises(RoleCapabilityError, match="execute_tests"):
                assert_role_command_allowed(role, separated_command)

        assert classify_command(["env", "-u", "CI", "git", "diff"]) == shell
        assert classify_command(["env", "--", "git", "diff"]) == shell

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

    def test_rejects_non_role_binding(self) -> None:
        with pytest.raises(ResearchRoleValidationError, match="role"):
            RoleBinding(
                role="verification_agent",  # type: ignore[arg-type]
                model_id="grok-4.6",
                provider="xai",
                agent_id="verifier-a",
            )


def test_package_exports_research_roles() -> None:
    import worktrees_hives as wh

    assert wh.ResearchRole is ResearchRole
    assert wh.RoleBinding is RoleBinding
    assert wh.v0_role is v0_role
    assert wh.assert_role_command_allowed is assert_role_command_allowed
