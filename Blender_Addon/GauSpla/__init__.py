bl_info = {
    "name": "GauSpla",
    "author": "OpenAI (Codex CLI) + You",
    "version": (0, 3, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > GauSpla",
    "description": "3D Gaussian Splat (.ply) viewer and sync bridge for Blender viewport",
    "category": "3D View",
}

from . import gsp_cache
from . import gsp_operators
from . import gsp_props
from . import gsp_sync
from . import gsp_ui
from . import gsp_viewport


def register():
    gsp_props.register()
    gsp_operators.register()
    gsp_ui.register()
    gsp_viewport.register()
    gsp_sync.register()


def unregister():
    gsp_sync.unregister()
    gsp_viewport.unregister()
    gsp_ui.unregister()
    gsp_operators.unregister()
    gsp_props.unregister()
    gsp_cache.clear()
