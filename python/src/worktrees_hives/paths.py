"""Platform-aware filesystem defaults for worktrees-hives.

Worktree root layout (override with ``WH_WORKTREE_BASE``)::

    {platform data}/worktrees-hives/worktrees

Must match Rust ``wh-core`` ``user_data_dir()`` + ``worktree_base_path()``
(``crates/wh-core/src/paths.rs``):

- Windows: ``%APPDATA%`` (fallback: ``%USERPROFILE%\\AppData\\Roaming``)
- macOS: ``~/Library/Application Support``
- Unix/Linux: ``$XDG_DATA_HOME`` or ``~/.local/share``

This is the single source of truth for the Python orchestrator default.
``claim`` and ``issue_to_pr`` must import from here — do not reimplement.
"""

from __future__ import annotations

import os
import sys

_WORKTREE_BASE_ENV = "WH_WORKTREE_BASE"


def _env_nonempty(name: str) -> str | None:
    """Return env var if set and non-empty (empty string counts as unset)."""
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def default_worktree_base(*, platform: str | None = None) -> str:
    """Return the default worktree root directory.

    Parameters
    ----------
    platform:
        Optional ``sys.platform`` override for tests (``win32``, ``darwin``,
        else Unix/XDG). Production callers leave this unset.
    """
    if override := _env_nonempty(_WORKTREE_BASE_ENV):
        return override

    plat = sys.platform if platform is None else platform
    return os.path.join(_user_data_dir(plat), "worktrees-hives", "worktrees")


def _user_data_dir(plat: str) -> str:
    """Mirror Rust ``wh-core::paths::user_data_dir`` platform branches."""
    if plat == "win32":
        # Prefer APPDATA (Roaming), not LOCALAPPDATA — must match wh-core.
        appdata = _env_nonempty("APPDATA")
        if appdata:
            return appdata
        profile = _env_nonempty("USERPROFILE")
        if profile:
            return os.path.join(profile, "AppData", "Roaming")
        return os.path.join(os.path.expanduser("~"), "AppData", "Roaming")

    if plat == "darwin":
        return os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
        )

    xdg = _env_nonempty("XDG_DATA_HOME")
    if xdg:
        return xdg
    return os.path.join(os.path.expanduser("~"), ".local", "share")
