from __future__ import annotations

import time

import bpy

from .gsp_bridge import canonical_watch_path, compute_file_signature, encode_file_signature, mark_sync_success, reload_gauspla_object, set_sync_status
from .gsp_props import OBJECT_SETTINGS_ATTR


_DEFAULT_POLL_INTERVAL = 0.25
_RUNTIME_CANDIDATES: dict[int, dict[str, object]] = {}


def _runtime_key(obj) -> int:
    return int(obj.as_pointer())


def clear_runtime_state(obj) -> None:
    if obj is None:
        return
    try:
        _RUNTIME_CANDIDATES.pop(_runtime_key(obj), None)
    except Exception:
        return


def _iter_linked_objects():
    for obj in bpy.data.objects:
        settings = getattr(obj, OBJECT_SETTINGS_ATTR, None)
        if settings is None or not bool(getattr(settings, "is_gauspla", False)):
            continue
        yield obj, settings


def _desired_poll_interval() -> float:
    interval = _DEFAULT_POLL_INTERVAL
    found = False
    for _obj, settings in _iter_linked_objects():
        if not settings.bridge_watch_path:
            continue
        if not bool(settings.bridge_auto_sync):
            continue
        try:
            candidate = max(0.1, float(settings.bridge_poll_interval))
        except Exception:
            candidate = _DEFAULT_POLL_INTERVAL
        if not found:
            interval = candidate
            found = True
        else:
            interval = min(interval, candidate)
    return interval


def _sync_timer() -> float:
    seen_keys: set[int] = set()
    try:
        for obj, settings in _iter_linked_objects():
            key = _runtime_key(obj)
            seen_keys.add(key)

            if not settings.bridge_watch_path:
                _RUNTIME_CANDIDATES.pop(key, None)
                continue

            if not bool(settings.bridge_auto_sync):
                _RUNTIME_CANDIDATES.pop(key, None)
                continue

            watch_path = canonical_watch_path(settings.bridge_watch_path)
            settings.bridge_watch_path = watch_path
            signature = compute_file_signature(watch_path)
            if signature is None:
                set_sync_status(settings, "MISSING_FILE")
                _RUNTIME_CANDIDATES.pop(key, None)
                continue

            signature_str = encode_file_signature(signature)
            if signature_str == str(settings.bridge_last_sig):
                if settings.bridge_status in {"WAITING_STABLE", "RELOADING", "MISSING_FILE"}:
                    set_sync_status(settings, "SYNCED")
                _RUNTIME_CANDIDATES.pop(key, None)
                continue

            candidate = _RUNTIME_CANDIDATES.get(key)
            if candidate is None or candidate.get("sig") != signature_str:
                _RUNTIME_CANDIDATES[key] = {"sig": signature_str, "seen": 1}
                set_sync_status(settings, "WAITING_STABLE")
                continue

            candidate["seen"] = int(candidate.get("seen", 0)) + 1
            if int(candidate["seen"]) < 2:
                set_sync_status(settings, "WAITING_STABLE")
                continue

            set_sync_status(settings, "RELOADING")
            try:
                reloaded_path, _data, applied_signature = reload_gauspla_object(obj, watch_path, update_filepath=True)
                settings.bridge_watch_path = reloaded_path
                mark_sync_success(settings, applied_signature)
            except Exception as exc:
                set_sync_status(settings, "RELOAD_FAILED", error=str(exc))
            finally:
                _RUNTIME_CANDIDATES.pop(key, None)

        stale_keys = [key for key in _RUNTIME_CANDIDATES if key not in seen_keys]
        for key in stale_keys:
            _RUNTIME_CANDIDATES.pop(key, None)
    except Exception as exc:
        print(f"[GauSpla] Sync timer error: {exc}")
    return _desired_poll_interval()


def register() -> None:
    _RUNTIME_CANDIDATES.clear()
    try:
        is_registered = bpy.app.timers.is_registered(_sync_timer)
    except Exception:
        is_registered = False
    if not is_registered:
        bpy.app.timers.register(_sync_timer, first_interval=_DEFAULT_POLL_INTERVAL, persistent=True)


def unregister() -> None:
    try:
        if bpy.app.timers.is_registered(_sync_timer):
            bpy.app.timers.unregister(_sync_timer)
    except Exception:
        pass
    _RUNTIME_CANDIDATES.clear()
