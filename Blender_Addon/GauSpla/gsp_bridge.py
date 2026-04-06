from __future__ import annotations

import os
import time

import bpy

from . import gsp_cache
from . import gsp_viewport
from .gsp_props import OBJECT_SETTINGS_ATTR


def canonical_watch_path(path: str) -> str:
    if not path:
        return ""
    try:
        abs_path = bpy.path.abspath(path)
    except Exception:
        abs_path = path
    return gsp_cache.canonical_path(abs_path)


def compute_file_signature(path: str) -> tuple[str, int, int] | None:
    cpath = canonical_watch_path(path)
    if not cpath:
        return None
    try:
        stat = os.stat(cpath)
    except OSError:
        return None
    mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
    return (cpath, int(stat.st_size), int(mtime_ns))


def encode_file_signature(signature: tuple[str, int, int] | None) -> str:
    if signature is None:
        return ""
    return f"{signature[0]}|{signature[1]}|{signature[2]}"


def format_reload_timestamp(timestamp: float) -> str:
    if float(timestamp) <= 0.0:
        return "Never"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(timestamp)))
    except Exception:
        return "Unknown"


def tag_view3d_redraw() -> None:
    try:
        wm = bpy.context.window_manager
    except Exception:
        wm = None
    if wm is None:
        return
    for window in wm.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def set_sync_status(settings, status: str, *, error: str | None = None) -> None:
    settings.bridge_status = status
    if error is not None:
        settings.bridge_last_error = str(error)


def mark_sync_success(settings, signature: str) -> None:
    settings.bridge_last_sig = signature
    settings.bridge_last_reload_ts = time.time()
    settings.bridge_last_error = ""
    settings.bridge_status = "SYNCED"


def clear_sync_link(settings) -> None:
    settings.bridge_watch_path = ""
    settings.bridge_auto_sync = False
    settings.bridge_last_sig = ""
    settings.bridge_last_reload_ts = 0.0
    settings.bridge_last_error = ""
    settings.bridge_status = "NOT_LINKED"


def reload_gauspla_object(obj, filepath: str, *, update_filepath: bool = True):
    settings = getattr(obj, OBJECT_SETTINGS_ATTR, None)
    if settings is None or not bool(getattr(settings, "is_gauspla", False)):
        raise RuntimeError("Object is not a GauSpla object")

    cpath = canonical_watch_path(filepath)
    if not cpath:
        raise RuntimeError("No PLY file set")
    if not os.path.exists(cpath):
        raise RuntimeError(f"File not found: {cpath}")

    matrix_world = obj.matrix_world.copy()
    data = gsp_cache.reload_if_valid(cpath, max_points=settings.max_points)
    if data.error or data.point_count == 0:
        raise RuntimeError(data.error or "Empty PLY")

    if update_filepath:
        settings.filepath = cpath
    obj.matrix_world = matrix_world
    gsp_viewport.reset_runtime_lod(obj)
    tag_view3d_redraw()
    return cpath, data, encode_file_signature(compute_file_signature(cpath))
