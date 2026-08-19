"""Public API for 3D k3d animations."""

from __future__ import annotations

from .k3d_anim import (
    ChunkedAnimator,
    ChunkedPlayer,
    FrameBinding,
    K3DAnimator,
    Player,
    StatefulPlayer,
    axis_aligned_plane,
    make_chunked_player,
    make_player,
    make_stateful_player,
    set_camera_overhead,
    set_camera_side,
)
from .k3d_helpers import make_dark_plot, surface_from_function

__all__ = [
    "ChunkedAnimator",
    "ChunkedPlayer",
    "FrameBinding",
    "K3DAnimator",
    "Player",
    "StatefulPlayer",
    "axis_aligned_plane",
    "make_chunked_player",
    "make_dark_plot",
    "make_player",
    "make_stateful_player",
    "set_camera_overhead",
    "set_camera_side",
    "surface_from_function",
]
