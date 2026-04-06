from __future__ import annotations

import math
import os

import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from . import gsp_cache
from . import gsp_sync
from .gsp_bridge import canonical_watch_path, compute_file_signature, encode_file_signature, mark_sync_success, reload_gauspla_object, set_sync_status
from .gsp_props import OBJECT_SETTINGS_ATTR, apply_look_preset, ensure_look_preset_initialized


def _active_gauspla_object(context):
    obj = context.object
    if obj is None or not hasattr(obj, OBJECT_SETTINGS_ATTR) or not obj.gau_spla_lfs.is_gauspla:
        return None
    _ensure_default_orientation(obj)
    ensure_look_preset_initialized(obj.gau_spla_lfs)
    return obj


def _ensure_default_orientation(obj) -> None:
    settings = getattr(obj, OBJECT_SETTINGS_ATTR, None)
    if settings is None:
        return
    if bool(getattr(settings, "axis_fix_initialized", False)):
        return

    rot = obj.rotation_euler
    tol = math.radians(1.0)
    if abs(float(rot.x) + (math.pi * 0.5)) <= tol and abs(float(rot.y)) <= tol and abs(float(rot.z)) <= tol:
        settings.axis_fix_initialized = True
        return
    if abs(float(rot.x)) <= tol and abs(float(rot.y)) <= tol and abs(float(rot.z)) <= tol:
        rot.x -= math.pi * 0.5
        settings.axis_fix_initialized = True
        return
    settings.axis_fix_initialized = True


def _set_reference_plus_defaults(settings) -> None:
    settings.render_mode = "REFERENCE_PLUS"
    settings.color_match = "NEUTRAL"
    settings.use_scene_depth_cull = True
    settings.auto_lod = False
    settings.draw_stride = 1
    settings.max_point_size = 224
    settings.sigma_clip = 2.35
    settings.min_pixel_cov = 0.02
    settings.allow_splat_distortion = False


def _pick_public_mode(point_count: int) -> str:
    del point_count
    return "REFERENCE_PLUS"


def _apply_mode_defaults(settings, mode: str) -> None:
    del mode
    _set_reference_plus_defaults(settings)


def _mark_linked_state(settings, watch_path: str) -> None:
    settings.bridge_watch_path = watch_path
    settings.bridge_last_error = ""
    if watch_path:
        signature = compute_file_signature(watch_path)
        if signature is not None:
            settings.bridge_last_sig = encode_file_signature(signature)
        settings.bridge_status = "LINKED"
    else:
        settings.bridge_last_sig = ""
        settings.bridge_status = "NOT_LINKED"


def _sync_success_for_path(settings, watch_path: str, signature: str) -> None:
    settings.bridge_watch_path = watch_path
    mark_sync_success(settings, signature)


def _set_linked_source(settings, watch_path: str) -> None:
    watch_path = canonical_watch_path(watch_path)
    _mark_linked_state(settings, watch_path)


def _spawn_gauspla_object(context, path: str):
    obj = bpy.data.objects.new(name=os.path.splitext(os.path.basename(path))[0], object_data=None)
    obj.empty_display_type = "ARROWS"
    context.collection.objects.link(obj)
    obj.location = context.scene.cursor.location
    obj.gau_spla_lfs.is_gauspla = True
    obj.gau_spla_lfs.filepath = path
    obj.gau_spla_lfs.enabled = True
    apply_look_preset(obj.gau_spla_lfs, "REFERENCE")
    _ensure_default_orientation(obj)
    data = gsp_cache.load(path, max_points=obj.gau_spla_lfs.max_points, force=True)
    _apply_mode_defaults(obj.gau_spla_lfs, _pick_public_mode(int(data.point_count)))
    return obj, data


def _finalize_import(context, path: str, *, linked: bool):
    obj, data = _spawn_gauspla_object(context, path)
    if linked:
        _set_linked_source(obj.gau_spla_lfs, path)
    context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj, data


class GAUSPLA_OT_import_linked_ply(Operator, ImportHelper):
    bl_idname = "gauspla_lfs.import_linked_ply"
    bl_label = "Import Linked PLY"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={"HIDDEN"})

    def execute(self, context):
        path = bpy.path.abspath(self.filepath)
        if not os.path.exists(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}

        obj, data = _finalize_import(context, path, linked=True)
        if data.error:
            self.report({"ERROR"}, f"Load failed: {data.error}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Loaded {data.point_count} points as linked source")
        return {"FINISHED"}


class GAUSPLA_OT_link_watch_file(Operator, ImportHelper):
    bl_idname = "gauspla_lfs.link_watch_file"
    bl_label = "Relink Watched Export"
    bl_options = {"REGISTER"}

    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={"HIDDEN"})

    def execute(self, context):
        obj = _active_gauspla_object(context)
        if obj is None:
            self.report({"ERROR"}, "Select a GauSpla object")
            return {"CANCELLED"}

        settings = obj.gau_spla_lfs
        watch_path = canonical_watch_path(self.filepath)
        if not watch_path:
            self.report({"ERROR"}, "No watched file selected")
            return {"CANCELLED"}

        current_path = canonical_watch_path(settings.filepath)
        gsp_sync.clear_runtime_state(obj)
        settings.bridge_watch_path = watch_path
        settings.bridge_last_error = ""

        if current_path and current_path == watch_path:
            _mark_linked_state(settings, watch_path)
            self.report({"INFO"}, "Watched export linked to current source file")
            return {"FINISHED"}

        set_sync_status(settings, "RELOADING", error="")
        try:
            reloaded_path, data, signature = reload_gauspla_object(obj, watch_path, update_filepath=True)
            _sync_success_for_path(settings, reloaded_path, signature)
            self.report({"INFO"}, f"Linked and loaded {data.point_count} points")
        except Exception as e:
            set_sync_status(settings, "RELOAD_FAILED", error=str(e))
            self.report({"WARNING"}, f"Link saved, but initial reload failed: {e}")
        return {"FINISHED"}


class GAUSPLA_OT_reload_linked_now(Operator):
    bl_idname = "gauspla_lfs.reload_linked_now"
    bl_label = "Reload Linked File Now"
    bl_options = {"REGISTER"}

    def execute(self, context):
        obj = _active_gauspla_object(context)
        if obj is None:
            self.report({"ERROR"}, "Select a GauSpla object")
            return {"CANCELLED"}
        settings = obj.gau_spla_lfs
        watch_path = canonical_watch_path(settings.bridge_watch_path)
        if not watch_path:
            self.report({"ERROR"}, "No watched export linked")
            return {"CANCELLED"}

        gsp_sync.clear_runtime_state(obj)
        set_sync_status(settings, "RELOADING", error="")
        try:
            reloaded_path, data, signature = reload_gauspla_object(obj, watch_path, update_filepath=True)
        except Exception as e:
            set_sync_status(settings, "RELOAD_FAILED", error=str(e))
            self.report({"ERROR"}, f"Reload failed: {e}")
            return {"CANCELLED"}

        _sync_success_for_path(settings, reloaded_path, signature)
        self.report({"INFO"}, f"Reloaded {data.point_count} points")
        return {"FINISHED"}


class GAUSPLA_OT_clear_link(Operator):
    bl_idname = "gauspla_lfs.clear_link"
    bl_label = "Clear Watched Export Link"
    bl_options = {"REGISTER"}

    def execute(self, context):
        obj = _active_gauspla_object(context)
        if obj is None:
            self.report({"ERROR"}, "Select a GauSpla object")
            return {"CANCELLED"}

        gsp_sync.clear_runtime_state(obj)
        settings = obj.gau_spla_lfs
        settings.bridge_watch_path = ""
        settings.bridge_auto_sync = False
        settings.bridge_last_sig = ""
        settings.bridge_last_reload_ts = 0.0
        settings.bridge_last_error = ""
        settings.bridge_status = "NOT_LINKED"
        self.report({"INFO"}, "Watched export link cleared")
        return {"FINISHED"}


class GAUSPLA_OT_set_display_mode(Operator):
    bl_idname = "gauspla_lfs.set_display_mode"
    bl_label = "Set Display Mode"
    bl_options = {"REGISTER"}

    mode: bpy.props.EnumProperty(
        name="Display Mode",
        items=(
            ("SPLATS", "Splats", "Render gaussian splats"),
            ("POINTS", "Points", "Render point preview"),
        ),
        default="SPLATS",
    )

    def execute(self, context):
        obj = _active_gauspla_object(context)
        if obj is None:
            self.report({"ERROR"}, "Select a GauSpla object")
            return {"CANCELLED"}

        settings = obj.gau_spla_lfs
        if self.mode == "POINTS":
            settings.enabled = False
            settings.point_preview = True
        else:
            settings.enabled = True
            settings.point_preview = True
        return {"FINISHED"}


_CLASSES = (
    GAUSPLA_OT_import_linked_ply,
    GAUSPLA_OT_link_watch_file,
    GAUSPLA_OT_reload_linked_now,
    GAUSPLA_OT_clear_link,
    GAUSPLA_OT_set_display_mode,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
