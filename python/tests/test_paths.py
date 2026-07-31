"""Tests for shared platform worktree path defaults.

Parameterized platform overrides ensure the macOS / Windows / XDG branches
run on every CI OS (linux, macos, windows runners all exercise all branches).

Windows selection must match Rust ``wh-core`` (APPDATA / Roaming, not Local).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from worktrees_hives import claim, issue_to_pr
from worktrees_hives.paths import default_worktree_base


class TestDefaultWorktreeBase:
    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WH_WORKTREE_BASE", "/tmp/custom-hive-wt")
        assert default_worktree_base() == "/tmp/custom-hive-wt"
        # Platform override must not override an explicit env base.
        assert default_worktree_base(platform="darwin") == "/tmp/custom-hive-wt"

    def test_darwin_branch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("WH_WORKTREE_BASE", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        # expanduser("~") on some platforms uses getpwuid; force via HOME.
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(home) if p == "~" else p)

        result = default_worktree_base(platform="darwin")
        expected = os.path.join(
            str(home),
            "Library",
            "Application Support",
            "worktrees-hives",
            "worktrees",
        )
        assert result == expected
        assert "Library" in Path(result).parts
        assert "Application Support" in Path(result).parts

    def test_win32_prefers_appdata_over_localappdata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rust wh-core uses APPDATA (Roaming); must not pick LOCALAPPDATA."""
        monkeypatch.delenv("WH_WORKTREE_BASE", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\ci\AppData\Local")
        monkeypatch.setenv("APPDATA", r"C:\Users\ci\AppData\Roaming")

        result = default_worktree_base(platform="win32")
        assert result == os.path.join(
            r"C:\Users\ci\AppData\Roaming", "worktrees-hives", "worktrees"
        )
        assert "Local" not in Path(result).parts

    def test_win32_appdata_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WH_WORKTREE_BASE", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.setenv("APPDATA", r"C:\Users\ci\AppData\Roaming")

        result = default_worktree_base(platform="win32")
        assert result == os.path.join(
            r"C:\Users\ci\AppData\Roaming", "worktrees-hives", "worktrees"
        )

    def test_win32_falls_back_to_userprofile_roaming(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No APPDATA must not fall through to darwin/Unix paths."""
        monkeypatch.delenv("WH_WORKTREE_BASE", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.delenv("APPDATA", raising=False)
        monkeypatch.setenv("USERPROFILE", r"C:\Users\ci")

        result = default_worktree_base(platform="win32")
        assert result == os.path.join(
            r"C:\Users\ci", "AppData", "Roaming", "worktrees-hives", "worktrees"
        )
        assert "Library" not in result
        assert ".local" not in result
        assert "Local" not in Path(result).parts

    def test_win32_ignores_empty_appdata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WH_WORKTREE_BASE", raising=False)
        monkeypatch.setenv("APPDATA", "")
        monkeypatch.setenv("USERPROFILE", r"C:\Users\ci")

        result = default_worktree_base(platform="win32")
        assert result == os.path.join(
            r"C:\Users\ci", "AppData", "Roaming", "worktrees-hives", "worktrees"
        )

    def test_xdg_data_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WH_WORKTREE_BASE", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", "/var/data")

        result = default_worktree_base(platform="linux")
        assert result == os.path.join("/var/data", "worktrees-hives", "worktrees")

    def test_unix_fallback_without_xdg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("WH_WORKTREE_BASE", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(home) if p == "~" else p)

        result = default_worktree_base(platform="linux")
        expected = os.path.join(str(home), ".local", "share", "worktrees-hives", "worktrees")
        assert result == expected

    @pytest.mark.parametrize(
        "platform",
        ["win32", "darwin", "linux"],
        ids=["win32", "darwin", "linux"],
    )
    def test_all_platform_branches_on_every_runner(
        self, monkeypatch: pytest.MonkeyPatch, platform: str
    ) -> None:
        """macOS/Windows/XDG logic must run on all three CI OS matrices."""
        monkeypatch.delenv("WH_WORKTREE_BASE", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Local")
        monkeypatch.setenv("APPDATA", r"C:\Roaming")
        home = "/fake-home"
        monkeypatch.setattr(os.path, "expanduser", lambda p: home if p == "~" else p)

        result = default_worktree_base(platform=platform)
        if platform == "win32":
            # APPDATA wins even when LOCALAPPDATA is set (Rust parity).
            assert result == os.path.join(r"C:\Roaming", "worktrees-hives", "worktrees")
        elif platform == "darwin":
            assert "Application Support" in result
            assert result.endswith(os.path.join("worktrees-hives", "worktrees")) or result.replace(
                "\\", "/"
            ).endswith("worktrees-hives/worktrees")
            assert result.startswith(os.path.join(home, "Library"))
        else:
            assert ".local" in Path(result).parts
            assert result == os.path.join(home, ".local", "share", "worktrees-hives", "worktrees")


class TestSharedImport:
    """claim and issue_to_pr must share one implementation (no drift)."""

    def test_claim_and_issue_to_pr_use_same_callable(self) -> None:
        assert claim._default_worktree_base is default_worktree_base
        assert issue_to_pr._default_worktree_base is default_worktree_base

    def test_claim_manager_default_factory_uses_shared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WH_WORKTREE_BASE", "/tmp/shared-factory-base")
        from unittest.mock import MagicMock

        mgr = claim.ClaimManager(wh_client=MagicMock())
        assert mgr.worktree_base == os.path.abspath("/tmp/shared-factory-base")
