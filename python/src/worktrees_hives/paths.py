"""Platform-aware filesystem defaults for worktrees-hives.

Worktree root layout (override with ``WH_WORKTREE_BASE``)::

    {platform data}/worktrees-hives/worktrees

Must match Rust ``wh-core`` ``user_data_dir()`` + ``worktree_base_path()``
(``crates/wh-core/src/paths.rs``):

- Windows: ``%APPDATA%`` → ``%USERPROFILE%\\AppData\\Roaming`` → temp dir
- macOS: ``~/Library/Application Support`` → temp dir
- Unix/Linux: ``$XDG_DATA_HOME`` → ``~/.local/share`` → temp dir

This is the single source of truth for the Python orchestrator default.
``claim`` and ``issue_to_pr`` must import from here — do not reimplement.
"""

from __future__ import annotations

import os
import sys
import tempfile

_WORKTREE_BASE_ENV = "WH_WORKTREE_BASE"


def _env_nonempty(name: str) -> str | None:
    """Return env var if set and non-empty (empty string counts as unset)."""
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def _home_dir() -> str | None:
    """Return the user's home directory, or ``None`` if it cannot be resolved."""
    home = os.path.expanduser("~")
    if home == "~" or not os.path.isabs(home):
        return None
    return home


def user_data_dir(*, platform: str | None = None) -> str:
    """Return the platform user-data directory, mirroring Rust ``wh-core``.

    Parameters
    ----------
    platform:
        Optional ``sys.platform`` override for tests (``win32``, ``darwin``,
        else Unix/XDG). Production callers leave this unset.
    """
    plat = sys.platform if platform is None else platform

    if plat == "win32":
        # Prefer APPDATA (Roaming), not LOCALAPPDATA — must match wh-core.
        appdata = _env_nonempty("APPDATA")
        if appdata:
            return appdata
        profile = _env_nonempty("USERPROFILE")
        if profile:
            return os.path.join(profile, "AppData", "Roaming")
        return tempfile.gettempdir()

    if plat == "darwin":
        home = _home_dir()
        if home is not None:
            return os.path.join(home, "Library", "Application Support")
        return tempfile.gettempdir()

    xdg = _env_nonempty("XDG_DATA_HOME")
    if xdg:
        return xdg
    home = _home_dir()
    if home is not None:
        return os.path.join(home, ".local", "share")
    return tempfile.gettempdir()


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

    return os.path.join(user_data_dir(platform=platform), "worktrees-hives", "worktrees")
