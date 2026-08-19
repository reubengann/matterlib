import matterlib
from matterlib import anim2d, anim3d
from matterlib.canvas2d_anim import Canvas2DAnimator, Canvas2DPlayer, ParamSpec
from matterlib.k3d_anim import K3DAnimator, make_player
from matterlib.k3d_helpers import make_dark_plot, surface_from_function


def test_root_exports_module_namespaces() -> None:
    assert matterlib.__all__ == ["anim2d", "anim3d", "spp"]
    assert matterlib.anim2d is anim2d
    assert matterlib.anim3d is anim3d


def test_anim2d_facade_exports_existing_implementations() -> None:
    assert set(anim2d.__all__) == {
        "BoundParam",
        "Canvas2DAnimator",
        "Canvas2DPlayer",
        "ParamSpec",
        "RepeatedTimer",
    }
    assert anim2d.Canvas2DAnimator is Canvas2DAnimator
    assert anim2d.Canvas2DPlayer is Canvas2DPlayer
    assert anim2d.ParamSpec is ParamSpec


def test_anim3d_facade_exports_existing_implementations() -> None:
    assert set(anim3d.__all__) == {
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
    }
    assert anim3d.K3DAnimator is K3DAnimator
    assert anim3d.make_player is make_player
    assert anim3d.make_dark_plot is make_dark_plot
    assert anim3d.surface_from_function is surface_from_function


def test_animation_names_are_not_exported_from_package_root() -> None:
    removed_names = (
        "BoundParam",
        "Canvas2DAnimator",
        "Canvas2DPlayer",
        "ChunkedAnimator",
        "ChunkedPlayer",
        "FrameBinding",
        "Player",
        "ParamSpec",
        "RepeatedTimer",
        "K3DAnimator",
        "StatefulPlayer",
        "axis_aligned_plane",
        "make_chunked_player",
        "make_stateful_player",
        "make_player",
        "make_dark_plot",
        "set_camera_overhead",
        "set_camera_side",
        "surface_from_function",
    )

    for name in removed_names:
        assert not hasattr(matterlib, name)
