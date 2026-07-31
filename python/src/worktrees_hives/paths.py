"""Platform-aware filesystem defaults for worktrees-hives.

Worktree root layout (override with ``WH_WORKTREE_BASE``)::

    {platform data}/worktrees-hives/worktrees

This is the single source of truth for the Python orchestrator default.
``claim`` and ``issue_to_pr`` must import from here — do not reimplement.
"""

from __future__ import annotations

import os
import sys

_WORKTREE_BASE_ENV = "WH_WORKTREE_BASE"


def default_worktree_base(*, platform: str | None = None) -> str:
    """Return the default worktree root directory.

    Parameters
    ----------
    platform:
        Optional ``sys.platform`` override for tests (``win32``, ``darwin``,
        else Unix/XDG). Production callers leave this unset.
    """
    if override := os.environ.get(_WORKTREE_BASE_ENV):
        return override

    plat = sys.platform if platform is None else platform

    if plat == "win32":
        local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if local:
            return os.path.join(local, "worktrees-hives", "worktrees")
    if plat == "darwin":
        return os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            "worktrees-hives",
            "worktrees",
        )
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return os.path.join(xdg, "worktrees-hives", "worktrees")
    return os.path.join(
        os.path.expanduser("~"),
        ".local",
        "share",
        "worktrees-hives",
        "worktrees",
    )
