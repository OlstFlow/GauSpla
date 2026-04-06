from __future__ import annotations

from .plugin import BlenderSyncPlugin

_PLUGIN = BlenderSyncPlugin()


def on_load() -> None:
    _PLUGIN.on_load()


def on_unload() -> None:
    _PLUGIN.on_unload()
