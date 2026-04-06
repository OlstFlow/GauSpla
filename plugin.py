from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import lichtfeld as lf

try:
    from lfs_plugins.operators import Operator
    from lfs_plugins.tool_defs.definition import ToolDef
    from lfs_plugins.tools import ToolRegistry
    from lfs_plugins.ui.state import AppState
except Exception:  # pragma: no cover - host-dependent fallback
    Operator = None
    ToolDef = None
    ToolRegistry = None
    AppState = None


PLUGIN_NAME = "gauspla_blender_sync"
PLUGIN_PATH = Path(__file__).resolve().parent
STATE_PATH = PLUGIN_PATH / "state.json"
LOG_PATH = PLUGIN_PATH / "sync.log"
STATE_VERSION = 1
SYNC_ICON_NAME = "sync"
LINK_ICON_NAME = "link"
PLY_FORMAT = 0
STATUS_TIMEOUT = 8.0
SYNC_TOOL_ID = f"{PLUGIN_NAME}.sync_tool"
DRAW_HANDLER_ID = f"{PLUGIN_NAME}.tick"
_PLUGIN_INSTANCE: "BlenderSyncPlugin | None" = None


def _canonical_path(value: str) -> str:
    if not value:
        return ""
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except Exception:
        try:
            return str(Path(value).expanduser())
        except Exception:
            return value


def _file_signature(path_str: str) -> tuple[int, int] | None:
    try:
        stat = Path(path_str).stat()
    except OSError:
        return None
    return (int(stat.st_size), int(stat.st_mtime_ns))


def _format_timestamp(timestamp: float) -> str:
    if timestamp <= 0.0:
        return "never"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    except Exception:
        return "unknown"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class LinkRecord:
    path: str
    last_sync_scene_generation: int = -1
    last_sync_time: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LinkRecord | None":
        path = _canonical_path(str(value.get("path", "")))
        if not path:
            return None
        return cls(
            path=path,
            last_sync_scene_generation=_safe_int(
                value.get("last_sync_scene_generation"),
                -1,
            ),
            last_sync_time=_safe_float(value.get("last_sync_time"), 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "last_sync_scene_generation": int(self.last_sync_scene_generation),
            "last_sync_time": float(self.last_sync_time),
        }


@dataclass
class ExportJob:
    scene_key: str
    scene_generation: int
    node_name: str
    path: str
    sh_degree: int
    before_signature: tuple[int, int] | None = None


class LinkStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._state: dict[str, Any] = {
            "version": STATE_VERSION,
            "scene_links": {},
        }

    def load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            self._state = {
                "version": STATE_VERSION,
                "scene_links": {},
            }
            return

        scene_links = raw.get("scene_links")
        if not isinstance(scene_links, dict):
            scene_links = {}

        normalized: dict[str, dict[str, Any]] = {}
        for scene_key, entries in scene_links.items():
            if not isinstance(scene_key, str) or not isinstance(entries, dict):
                continue
            scene_bucket: dict[str, Any] = {}
            for node_name, payload in entries.items():
                if not isinstance(node_name, str) or not isinstance(payload, dict):
                    continue
                record = LinkRecord.from_dict(payload)
                if record is not None:
                    scene_bucket[node_name] = record.to_dict()
            if scene_bucket:
                normalized[scene_key] = scene_bucket

        self._state = {
            "version": STATE_VERSION,
            "scene_links": normalized,
        }

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._state, indent=2, sort_keys=True)
        self._path.write_text(payload + "\n", encoding="utf-8")

    def get_link(self, scene_key: str, node_name: str) -> LinkRecord | None:
        if not scene_key or not node_name:
            return None
        payload = (
            self._state.get("scene_links", {})
            .get(scene_key, {})
            .get(node_name)
        )
        if not isinstance(payload, dict):
            return None
        return LinkRecord.from_dict(payload)

    def set_link(self, scene_key: str, node_name: str, path: str) -> LinkRecord:
        if not scene_key:
            raise ValueError("scene_key is required")
        if not node_name:
            raise ValueError("node_name is required")

        record = self.get_link(scene_key, node_name)
        if record is None:
            record = LinkRecord(path=_canonical_path(path))
        else:
            record.path = _canonical_path(path)

        if not record.path:
            raise ValueError("path is required")

        scene_bucket = self._state.setdefault("scene_links", {}).setdefault(scene_key, {})
        scene_bucket[node_name] = record.to_dict()
        self.save()
        return record

    def clear_link(self, scene_key: str, node_name: str) -> bool:
        scene_bucket = self._state.get("scene_links", {}).get(scene_key)
        if not isinstance(scene_bucket, dict):
            return False
        removed = scene_bucket.pop(node_name, None) is not None
        if removed:
            if not scene_bucket:
                self._state.get("scene_links", {}).pop(scene_key, None)
            self.save()
        return removed

    def mark_synced(
        self,
        scene_key: str,
        node_name: str,
        generation: int,
        when: float | None = None,
    ) -> None:
        record = self.get_link(scene_key, node_name)
        if record is None:
            return
        record.last_sync_scene_generation = int(generation)
        record.last_sync_time = float(time.time() if when is None else when)
        self._state["scene_links"][scene_key][node_name] = record.to_dict()
        self.save()


class BlenderSyncPlugin:
    def __init__(self) -> None:
        self._store = LinkStore(STATE_PATH)
        self._sync_icon_id = 0
        self._link_icon_id = 0
        self._status_message = "Select a splat node to sync."
        self._status_level = "info"
        self._status_until = 0.0
        self._queue: deque[ExportJob] = deque()
        self._active_job: ExportJob | None = None
        self._batch_total = 0
        self._batch_done = 0
        self._batch_failed: list[str] = []
        self._batch_unlinked: list[str] = []
        self._batch_started_at = 0.0
        self._last_non_sync_tool_id = "builtin.select"

    def on_load(self) -> None:
        global _PLUGIN_INSTANCE
        _PLUGIN_INSTANCE = self
        self._store.load()
        self._ensure_icons_loaded()
        self._register_toolbar_tool()
        lf.ui.add_hook("tools", "transform", self.draw_tools_hook, "append")
        self._register_background_tick()
        self._enable_load_on_startup()
        self._log("plugin loaded")
        self._set_status("Blender sync ready.", level="success")

    def on_unload(self) -> None:
        global _PLUGIN_INSTANCE
        try:
            lf.ui.remove_hook("tools", "transform", self.draw_tools_hook)
        except Exception:
            pass

        self._unregister_background_tick()
        self._unregister_toolbar_tool()

        try:
            lf.ui.free_plugin_icons(PLUGIN_NAME)
        except Exception:
            pass

        try:
            lf.ui.free_plugin_textures(PLUGIN_NAME)
        except Exception:
            pass

        self._queue.clear()
        self._active_job = None
        self._log("plugin unloaded")
        _PLUGIN_INSTANCE = None

    def draw_tools_hook(self, layout) -> None:
        try:
            self._tick()
            scene = lf.get_scene()
            if scene is None:
                return

            splat_nodes = self._get_splat_nodes(scene)
            if not splat_nodes:
                return

            if not layout.collapsing_header("Blender Sync", default_open=True):
                return

            self._remember_active_tool()
            targets = self._resolve_target_nodes(scene, splat_nodes)
            scene_key = self._scene_key()
            current_generation = self._scene_generation()
            link_state = self._collect_link_state(scene_key, targets, current_generation)

            self._draw_sync_row(layout, link_state)
            self._draw_scope_summary(layout, scene_key, targets, link_state)
            self._draw_bind_controls(layout, scene_key, targets)
            self._draw_status(layout)
        except Exception as exc:
            self._set_status(f"Plugin error: {exc}", level="error", sticky=True)

    def trigger_toolbar_sync(self) -> None:
        self._queue_sync()

    def _draw_sync_row(self, layout, link_state: dict[str, Any]) -> None:
        button_size = self._toolbar_button_size()
        busy = self._is_export_busy()
        short_status = self._short_status(link_state)

        if self._sync_icon_id:
            if layout.toolbar_button(
                "gauspla_blender_sync.sync",
                self._sync_icon_id,
                (button_size, button_size),
                False,
                busy,
                "Sync linked splats to Blender",
            ):
                self._queue_sync()
        else:
            layout.begin_disabled(busy)
            try:
                if layout.button("Sync To Blender", (0.0, 0.0)):
                    self._queue_sync()
            finally:
                layout.end_disabled()

        layout.same_line()
        layout.text_colored(short_status, self._status_color(self._short_status_level(link_state)))

    def _draw_scope_summary(
        self,
        layout,
        scene_key: str,
        targets: list[Any],
        link_state: dict[str, Any],
    ) -> None:
        if not targets:
            layout.text_disabled("Scope: select one or more splat nodes.")
            return

        if len(targets) == 1:
            node_name = targets[0].name
            record = self._store.get_link(scene_key, node_name)
            if record is None:
                layout.text_disabled(f"Scope: {node_name} (unlinked)")
                return

            label = f"Scope: {node_name} -> {Path(record.path).name}"
            if self._link_icon_id:
                icon_size = max(14.0, self._toolbar_button_size() * 0.72)
                layout.image(self._link_icon_id, (icon_size, icon_size))
                layout.same_line()
            layout.label(label)

            if not Path(record.path).parent.exists():
                layout.text_colored("Linked folder is missing on disk.", self._status_color("warning"))
            else:
                layout.text_disabled(
                    "Last sync: "
                    f"{_format_timestamp(record.last_sync_time)}"
                )
            return

        layout.text_disabled(
            "Scope: "
            f"{len(targets)} selected, "
            f"{link_state['linked_count']} linked, "
            f"{link_state['unlinked_count']} unlinked"
        )

    def _draw_bind_controls(self, layout, scene_key: str, targets: list[Any]) -> None:
        single_target = len(targets) == 1
        current_file = self._current_scene_file()
        can_bind_current = single_target and current_file.lower().endswith(".ply")
        can_browse = single_target
        can_clear = bool(targets)

        layout.separator()

        layout.begin_disabled(not can_bind_current)
        try:
            if layout.small_button("Bind Selected To Current File"):
                self._bind_to_current_file(scene_key, targets[0])
        finally:
            layout.end_disabled()

        layout.same_line()

        layout.begin_disabled(not can_browse)
        try:
            if layout.small_button("Bind Selected To PLY"):
                self._browse_link(scene_key, targets[0])
        finally:
            layout.end_disabled()

        layout.same_line()

        layout.begin_disabled(not can_clear)
        try:
            if layout.small_button("Clear Link"):
                self._clear_links(scene_key, targets)
        finally:
            layout.end_disabled()

        if not single_target:
            layout.text_disabled("Binding is single-node only. Multi-select sync stays per-file.")
        elif not current_file.lower().endswith(".ply"):
            layout.text_disabled("Current scene file is not a .ply, so direct bind is unavailable.")

    def _draw_status(self, layout) -> None:
        layout.separator()
        layout.text_colored(self._status_message, self._status_color(self._status_level))

    def _tick(self) -> None:
        now = time.monotonic()
        if self._status_until and now > self._status_until and not self._is_export_busy():
            self._status_until = 0.0
            self._status_level = "info"
            self._status_message = "Ready."

        self._poll_export_queue()

    def _register_toolbar_tool(self) -> None:
        if ToolRegistry is None or ToolDef is None or Operator is None:
            self._set_status("Toolbar API unavailable; using panel-only mode.", level="warning", sticky=True)
            return

        try:
            lf.register_class(SyncToolbarOperator)
        except Exception:
            pass

        tool = ToolDef(
            id=SYNC_TOOL_ID,
            label="Blender Sync",
            icon=SYNC_ICON_NAME,
            group="transform",
            order=55,
            description="Sync linked splats to Blender",
            operator=SyncToolbarOperator._class_id(),
            plugin_name=PLUGIN_NAME,
            plugin_path=str(PLUGIN_PATH),
            action_only=True,
            poll=self._poll_toolbar_tool,
            selected=self._toolbar_selected,
        )
        ToolRegistry.register_tool(tool)

    def _register_background_tick(self) -> None:
        try:
            lf.add_draw_handler(DRAW_HANDLER_ID, self._background_tick, "POST_VIEW")
        except Exception as exc:
            self._log(f"background tick register failed: {exc}")

    def _unregister_background_tick(self) -> None:
        try:
            lf.remove_draw_handler(DRAW_HANDLER_ID)
        except Exception:
            pass

    def _background_tick(self, *args, **kwargs) -> None:
        del args, kwargs
        try:
            self._tick()
        except Exception as exc:
            self._log(f"background tick error: {exc}")

    def _unregister_toolbar_tool(self) -> None:
        if ToolRegistry is not None:
            try:
                ToolRegistry.unregister_tool(SYNC_TOOL_ID)
            except Exception:
                pass

        if Operator is not None:
            try:
                lf.unregister_class(SyncToolbarOperator)
            except Exception:
                pass

    @staticmethod
    def _poll_toolbar_tool(context: Any) -> bool:
        return bool(getattr(context, "has_scene", False))

    def _toolbar_selected(self, context: Any) -> bool:
        del context
        return self._is_export_busy()

    def _remember_active_tool(self) -> None:
        if ToolRegistry is None:
            return
        try:
            active_tool = lf.ui.get_active_tool() or ""
        except Exception:
            return
        if active_tool and active_tool != SYNC_TOOL_ID:
            self._last_non_sync_tool_id = active_tool

    def _restore_previous_tool(self) -> None:
        if ToolRegistry is None:
            return
        target_tool = self._last_non_sync_tool_id or "builtin.select"
        if target_tool == SYNC_TOOL_ID:
            target_tool = "builtin.select"
        try:
            ToolRegistry.set_active(target_tool)
        except Exception:
            try:
                ToolRegistry.clear_active()
                ToolRegistry.set_active("builtin.select")
            except Exception:
                pass

    def _enable_load_on_startup(self) -> None:
        try:
            prefs = lf.plugins.settings(PLUGIN_NAME)
            if not prefs.get("load_on_startup", False):
                prefs.set("load_on_startup", True)
        except Exception:
            pass

    def _queue_sync(self) -> None:
        if self._is_export_busy():
            self._log("sync ignored: export busy")
            self._set_status("A sync batch is already running.", level="warning")
            return

        scene = lf.get_scene()
        if scene is None:
            self._log("sync aborted: no scene loaded")
            self._set_status("No scene loaded.", level="error")
            return

        scene_key = self._scene_key()
        if not scene_key:
            self._log("sync aborted: empty scene key")
            self._set_status("Current scene has no canonical path yet.", level="error", sticky=True)
            return

        targets = self._resolve_target_nodes(scene, self._get_splat_nodes(scene))
        if not targets:
            self._log("sync aborted: no target nodes")
            self._set_status("Select splat nodes to sync.", level="warning")
            return

        self._auto_bind_single_target(scene_key, targets)

        current_generation = self._scene_generation()
        link_state = self._collect_link_state(scene_key, targets, current_generation)
        self._log(
            "queue sync: "
            f"scene='{scene_key}', "
            f"generation={current_generation}, "
            f"targets={[node.name for node in targets]}, "
            f"linked={link_state['linked_count']}, "
            f"unlinked={link_state['unlinked_count']}"
        )

        self._batch_unlinked = [node.name for node in link_state["unlinked"]]
        jobs = []
        path_errors: list[str] = []
        for node in link_state["linked"]:
            record = self._store.get_link(scene_key, node.name)
            if record is None:
                continue

            parent = Path(record.path).parent
            if not record.path.lower().endswith(".ply"):
                path_errors.append(f"{node.name}: linked path is not a PLY")
                continue
            if not parent.exists():
                path_errors.append(f"{node.name}: linked folder is missing")
                continue

            before_signature = _file_signature(record.path)
            jobs.append(
                ExportJob(
                    scene_key=scene_key,
                    scene_generation=current_generation,
                    node_name=node.name,
                    path=record.path,
                    sh_degree=self._node_sh_degree(node),
                    before_signature=before_signature,
                )
            )
            self._log(
                "job queued: "
                f"node='{node.name}', "
                f"path='{record.path}', "
                f"before={before_signature}"
            )

        self._batch_failed = path_errors
        self._batch_total = len(jobs)
        self._batch_done = 0
        self._queue = deque(jobs)
        self._active_job = None
        self._batch_started_at = time.perf_counter() if jobs else 0.0

        if not self._queue:
            if self._batch_unlinked:
                self._log(f"sync produced no jobs: unlinked={self._batch_unlinked}")
                self._set_status(
                    f"No linked exports queued. Unlinked: {', '.join(self._batch_unlinked)}.",
                    level="warning",
                )
            elif path_errors:
                self._log(f"sync produced no jobs: path_errors={path_errors}")
                self._set_status("; ".join(path_errors), level="error", sticky=True)
            else:
                self._log("sync produced no jobs: nothing linked")
                self._set_status("Nothing new to sync.", level="info")
            return

        self._set_status(
            f"Queued {self._batch_total} export(s).",
            level="info",
            sticky=True,
        )
        self._poll_export_queue()

    def _auto_bind_single_target(self, scene_key: str, targets: list[Any]) -> None:
        if len(targets) != 1:
            return
        if not scene_key.lower().endswith(".ply"):
            return

        node = targets[0]
        if self._store.get_link(scene_key, node.name) is not None:
            return

        try:
            self._store.set_link(scene_key, node.name, scene_key)
            self._log(f"auto-linked node='{node.name}' to '{scene_key}'")
            self._set_status(
                f"Auto-linked {node.name} to current PLY.",
                level="info",
            )
        except Exception as exc:
            self._set_status(f"Auto-link failed: {exc}", level="error", sticky=True)

    def _poll_export_queue(self) -> None:
        export_state = lf.ui.get_export_state()
        export_active = bool(export_state.get("active", False))

        if self._active_job is not None and not export_active:
            self._log(
                "export finished by state transition: "
                f"job='{self._active_job.node_name}', state={dict(export_state)}"
            )
            self._finish_active_job()

        if self._active_job is None and self._queue and not export_active:
            self._start_next_job()

    def _start_next_job(self) -> None:
        while self._queue:
            job = self._queue.popleft()
            try:
                lf.export_scene(
                    PLY_FORMAT,
                    job.path,
                    [job.node_name],
                    int(job.sh_degree),
                )
            except Exception as exc:
                self._log(f"export invoke failed for '{job.node_name}': {exc}")
                self._batch_failed.append(f"{job.node_name}: {exc}")
                continue

            self._active_job = job
            self._log(
                "export started: "
                f"node='{job.node_name}', "
                f"path='{job.path}', "
                f"sh_degree={job.sh_degree}, "
                f"before={job.before_signature}"
            )
            self._set_status(
                "Syncing "
                f"{job.node_name} "
                f"({self._batch_done + 1}/{self._batch_total})...",
                level="info",
                sticky=True,
            )
            return

        self._finalize_batch()

    def _finish_active_job(self) -> None:
        job = self._active_job
        self._active_job = None
        if job is None:
            return

        after_signature = _file_signature(job.path)
        self._log(
            "export finalized: "
            f"node='{job.node_name}', "
            f"path='{job.path}', "
            f"before={job.before_signature}, "
            f"after={after_signature}"
        )
        if after_signature is not None and after_signature != job.before_signature:
            self._store.mark_synced(
                job.scene_key,
                job.node_name,
                job.scene_generation,
            )
            self._batch_done += 1
            self._log(
                "export success: "
                f"node='{job.node_name}', generation={job.scene_generation}"
            )
        else:
            self._batch_failed.append(f"{job.node_name}: export did not update the file")
            self._log(f"export failed: node='{job.node_name}', signature unchanged")

        if self._queue:
            self._start_next_job()
            return

        self._finalize_batch()

    def _finalize_batch(self) -> None:
        if self._active_job is not None or self._queue:
            return

        if self._batch_total == 0 and not self._batch_failed and not self._batch_unlinked:
            return

        elapsed = 0.0
        if self._batch_started_at > 0.0:
            elapsed = max(0.0, time.perf_counter() - self._batch_started_at)

        parts: list[str] = []
        if self._batch_done:
            parts.append(f"Synced {self._batch_done}")
        if self._batch_failed:
            parts.append(f"failed {len(self._batch_failed)}")
        if self._batch_unlinked:
            parts.append(f"unlinked {len(self._batch_unlinked)}")

        if not parts:
            parts.append("Nothing new to sync")

        if self._batch_failed:
            detail = "; ".join(self._batch_failed[:3])
            message = f"{', '.join(parts)}. {detail}"
            level = "warning" if self._batch_done else "error"
        elif self._batch_unlinked:
            unlinked_names = ", ".join(self._batch_unlinked[:3])
            message = f"{', '.join(parts)}. Bind first: {unlinked_names}."
            level = "warning"
        else:
            message = f"{', '.join(parts)}."
            level = "success"

        if elapsed > 0.0 and self._batch_done:
            message = message[:-1] + f" in {elapsed:.1f}s."

        self._set_status(message, level=level, sticky=True)
        self._log(
            "batch finalized: "
            f"done={self._batch_done}, failed={self._batch_failed}, "
            f"unlinked={self._batch_unlinked}, elapsed={elapsed:.3f}s"
        )
        self._batch_total = 0
        self._batch_done = 0
        self._batch_failed = []
        self._batch_unlinked = []
        self._batch_started_at = 0.0

    def _bind_to_current_file(self, scene_key: str, node: Any) -> None:
        current_file = self._current_scene_file()
        if not current_file.lower().endswith(".ply"):
            self._set_status("Current scene file is not a .ply.", level="error")
            return

        self._store.set_link(scene_key, node.name, current_file)
        self._set_status(
            f"{node.name} linked to {Path(current_file).name}.",
            level="success",
        )

    def _browse_link(self, scene_key: str, node: Any) -> None:
        existing = self._store.get_link(scene_key, node.name)
        start_dir = ""
        if existing is not None:
            start_dir = str(Path(existing.path).parent)
        elif self._current_scene_file():
            start_dir = str(Path(self._current_scene_file()).parent)

        path = lf.ui.open_ply_file_dialog(start_dir)
        if not path:
            return

        linked_path = _canonical_path(path)
        if not linked_path.lower().endswith(".ply"):
            self._set_status("Only .ply targets are supported.", level="error")
            return

        self._store.set_link(scene_key, node.name, linked_path)
        self._set_status(
            f"{node.name} linked to {Path(linked_path).name}.",
            level="success",
        )

    def _clear_links(self, scene_key: str, targets: list[Any]) -> None:
        if not targets:
            self._set_status("Nothing selected to clear.", level="warning")
            return

        removed = 0
        for node in targets:
            removed += int(self._store.clear_link(scene_key, node.name))

        if removed:
            self._set_status(f"Cleared {removed} link(s).", level="success")
        else:
            self._set_status("Selected nodes were already unlinked.", level="info")

    def _collect_link_state(
        self,
        scene_key: str,
        targets: list[Any],
        current_generation: int,
    ) -> dict[str, Any]:
        linked: list[Any] = []
        unlinked: list[Any] = []
        stale_linked: list[Any] = []
        clean_linked: list[Any] = []

        for node in targets:
            record = self._store.get_link(scene_key, node.name)
            if record is None:
                unlinked.append(node)
                continue
            linked.append(node)
            if record.last_sync_scene_generation == current_generation:
                clean_linked.append(node)
            else:
                stale_linked.append(node)

        return {
            "linked": linked,
            "unlinked": unlinked,
            "stale_linked": stale_linked,
            "clean_linked": clean_linked,
            "linked_count": len(linked),
            "unlinked_count": len(unlinked),
        }

    def _resolve_target_nodes(self, scene: Any, splat_nodes: list[Any] | None = None) -> list[Any]:
        if splat_nodes is None:
            splat_nodes = self._get_splat_nodes(scene)

        by_name = {node.name: node for node in splat_nodes}
        selected = []
        try:
            for name in lf.get_selected_node_names():
                node = by_name.get(name)
                if node is not None:
                    selected.append(node)
        except Exception:
            selected = []

        if selected:
            return selected
        if len(splat_nodes) == 1:
            return list(splat_nodes)
        return []

    def _get_splat_nodes(self, scene: Any) -> list[Any]:
        nodes = []
        try:
            iterable = scene.get_nodes(lf.scene.NodeType.SPLAT)
        except TypeError:
            iterable = scene.get_nodes()
        except Exception:
            return nodes

        for node in iterable:
            try:
                if node.type == lf.scene.NodeType.SPLAT and node.gaussian_count > 0:
                    nodes.append(node)
            except Exception:
                continue
        return nodes

    def _scene_key(self) -> str:
        return _canonical_path(self._current_scene_file())

    def _current_scene_file(self) -> str:
        if AppState is not None:
            try:
                return _canonical_path(AppState.scene_path.value)
            except Exception:
                pass
        return ""

    def _scene_generation(self) -> int:
        if AppState is not None:
            try:
                return int(AppState.scene_generation.value)
            except Exception:
                pass
        try:
            return int(lf.get_scene_generation())
        except Exception:
            return 0

    def _node_sh_degree(self, node: Any) -> int:
        try:
            splat = node.splat_data()
            if splat is not None:
                return max(0, min(3, int(splat.active_sh_degree)))
        except Exception:
            pass
        return 3

    def _ensure_icons_loaded(self) -> None:
        if not self._sync_icon_id:
            try:
                self._sync_icon_id = lf.ui.load_plugin_icon(
                    SYNC_ICON_NAME,
                    str(PLUGIN_PATH),
                    PLUGIN_NAME,
                )
            except Exception:
                self._sync_icon_id = 0

        if not self._link_icon_id:
            try:
                self._link_icon_id = lf.ui.load_plugin_icon(
                    LINK_ICON_NAME,
                    str(PLUGIN_PATH),
                    PLUGIN_NAME,
                )
            except Exception:
                self._link_icon_id = 0

    def _toolbar_button_size(self) -> float:
        try:
            return max(18.0, float(lf.ui.theme().sizes.toolbar_button_size))
        except Exception:
            return 22.0

    def _is_export_busy(self) -> bool:
        if self._active_job is not None or self._queue:
            return True
        try:
            return bool(lf.ui.get_export_state().get("active", False))
        except Exception:
            return False

    def _short_status(self, link_state: dict[str, Any]) -> str:
        if self._is_export_busy():
            try:
                export_state = lf.ui.get_export_state()
                progress = max(0.0, min(1.0, float(export_state.get("progress", 0.0))))
            except Exception:
                progress = 0.0
            total = max(self._batch_total, self._batch_done + int(self._active_job is not None))
            active_index = self._batch_done + int(self._active_job is not None)
            if progress > 0.0:
                if total <= 1:
                    return f"sync {progress * 100:.0f}%"
                return f"sync {active_index}/{total} {progress * 100:.0f}%"
            return f"syncing {active_index}/{total}"

        linked_count = link_state["linked_count"]
        unlinked_count = link_state["unlinked_count"]

        if linked_count == 0 and unlinked_count == 0:
            return "select splats"
        if linked_count == 0:
            return "unlinked"
        if unlinked_count == 0 and linked_count == 1:
            return "linked"
        if unlinked_count == 0:
            return f"{linked_count} linked"
        return f"{linked_count} linked, {unlinked_count} unlinked"

    def _short_status_level(self, link_state: dict[str, Any]) -> str:
        if self._is_export_busy():
            return "info"
        if link_state["linked_count"] == 0 and link_state["unlinked_count"] > 0:
            return "warning"
        if link_state["unlinked_count"] > 0:
            return "warning"
        if len(link_state["stale_linked"]) == 0 and link_state["linked_count"] > 0:
            return "success"
        return "info"

    def _status_color(self, level: str) -> Any:
        try:
            palette = lf.ui.theme().palette
            return {
                "success": palette.success,
                "warning": palette.warning,
                "error": palette.error,
                "info": palette.info,
            }.get(level, palette.text)
        except Exception:
            return (1.0, 1.0, 1.0, 1.0)

    def _set_status(self, message: str, level: str = "info", sticky: bool = False) -> None:
        self._status_message = message
        self._status_level = level
        self._status_until = 0.0 if sticky else (time.monotonic() + STATUS_TIMEOUT)

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(f"[{timestamp}] {message}\n")
        except Exception:
            pass


class SyncToolbarOperator(Operator if Operator is not None else object):
    label = "Blender Sync"
    description = "Sync linked splats to Blender"

    @classmethod
    def poll(cls, context) -> bool:
        del context
        return _PLUGIN_INSTANCE is not None

    def execute(self, context) -> set:
        del context
        if _PLUGIN_INSTANCE is None:
            return {"CANCELLED"}
        _PLUGIN_INSTANCE.trigger_toolbar_sync()
        return {"FINISHED"}
