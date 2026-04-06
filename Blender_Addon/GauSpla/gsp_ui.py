from __future__ import annotations

import bpy
from bpy.types import Panel

from . import gsp_viewport
from .gsp_bridge import format_reload_timestamp
from .gsp_props import OBJECT_SETTINGS_ATTR, bridge_status_label, ensure_look_preset_initialized


def _foldout(parent, settings, prop_name: str, title: str):
    box = parent.box()
    row = box.row(align=True)
    is_open = bool(getattr(settings, prop_name))
    row.prop(settings, prop_name, text=title, icon=("TRIA_DOWN" if is_open else "TRIA_RIGHT"), emboss=False)
    return box, is_open


class VIEW3D_PT_gauspla_lfs(Panel):
    bl_label = "GauSpla"
    bl_idname = "VIEW3D_PT_gauspla_lfs"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GauSpla"

    def draw(self, context):
        layout = self.layout
        start_box = layout.box()
        start_box.label(text="Start", icon="IMPORT")
        start_box.operator("gauspla_lfs.import_linked_ply", icon="LINKED", text="Import Linked PLY")

        obj = context.object
        if obj is None:
            layout.separator()
            layout.label(text="Select a GauSpla object")
            return

        if not hasattr(obj, OBJECT_SETTINGS_ATTR) or not obj.gau_spla_lfs.is_gauspla:
            layout.separator()
            layout.label(text="Select a GauSpla object")
            return

        s = obj.gau_spla_lfs
        ensure_look_preset_initialized(s)
        err = gsp_viewport.get_last_draw_error()
        if err:
            box = layout.box()
            box.label(text="Viewport Error", icon="ERROR")
            box.label(text=str(err))

        sync_box, sync_open = _foldout(layout, s, "ui_show_sync", "Sync")
        if sync_open:
            watch_path = bpy.path.abspath(s.bridge_watch_path) if s.bridge_watch_path else ""
            path_row = sync_box.row()
            path_row.enabled = False
            path_row.prop(s, "bridge_watch_path", text="Linked File")

            auto_row = sync_box.row()
            auto_row.enabled = bool(watch_path)
            auto_row.prop(s, "bridge_auto_sync")

            poll_row = sync_box.row()
            poll_row.enabled = bool(watch_path) and bool(s.bridge_auto_sync)
            poll_row.prop(s, "bridge_poll_interval", slider=True)

            buttons = sync_box.row(align=True)
            buttons.operator("gauspla_lfs.link_watch_file", icon="FILE_FOLDER", text=("Relink" if watch_path else "Link"))
            reload_now = buttons.row(align=True)
            reload_now.enabled = bool(watch_path)
            reload_now.operator("gauspla_lfs.reload_linked_now", icon="FILE_REFRESH", text="Reload Now")
            clear_link = buttons.row(align=True)
            clear_link.enabled = bool(watch_path)
            clear_link.operator("gauspla_lfs.clear_link", icon="X", text="Clear Link")

            status_box = sync_box.box()
            status_box.label(text=f"Status: {bridge_status_label(s.bridge_status)}", icon="FILE_REFRESH")
            if s.bridge_last_reload_ts > 0.0:
                status_box.label(text=f"Last Reload: {format_reload_timestamp(s.bridge_last_reload_ts)}")
            if not watch_path:
                status_box.label(text="No linked source yet")

            if s.bridge_last_error:
                err_box = sync_box.box()
                err_box.alert = True
                err_box.label(text="Last Error", icon="ERROR")
                err_box.label(text=s.bridge_last_error)

        view_box, view_open = _foldout(layout, s, "ui_show_display", "View")
        if view_open:
            view_box.prop(s, "max_points")
            mode_row = view_box.row(align=True)
            op = mode_row.operator(
                "gauspla_lfs.set_display_mode",
                text="Splats",
                depress=bool(s.enabled),
            )
            op.mode = "SPLATS"
            op = mode_row.operator(
                "gauspla_lfs.set_display_mode",
                text="Points",
                depress=not bool(s.enabled),
            )
            op.mode = "POINTS"
            if not bool(s.enabled):
                view_box.prop(s, "disabled_point_size", text="Point Size")

        if bool(s.enabled):
            look_box, look_open = _foldout(layout, s, "ui_show_look", "Appearance")
            if look_open:
                look_box.prop(s, "color_gain")
                look_box.prop(s, "alpha_multiplier")
                look_box.prop(s, "color_saturation")


_CLASSES = (VIEW3D_PT_gauspla_lfs,)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
