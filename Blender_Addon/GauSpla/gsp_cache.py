from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

from .gsp_ply import GaussianPlyData, PlyError, PlyRawData, load_gaussian_ply, load_gaussian_ply_raw
from .gsp_utils import eval_sh_rgb


@dataclass
class GauSplaData:
    path: str
    point_count: int
    positions: object
    colors: object
    opacity_raw: object
    opacity_draw: object
    scale_raw: object
    rot_raw: object
    rot_draw: object
    sh_coeffs: object | None
    sh_degree: int
    color_source: str
    loaded_at: float
    error: str | None = None
    scale_log_upper: float | None = None
    filtered_count: int = 0
    gpu_batch: object | None = None
    gpu_batch_oit: object | None = None
    gpu_batch_sorted: object | None = None
    gpu_batch_refplus: object | None = None
    gpu_sort_key: object | None = None
    gpu_sort_last_time: float = 0.0
    gpu_refplus_key: object | None = None
    gpu_refplus_last_time: float = 0.0
    last_color_key: object | None = None
    pending_color_key: object | None = None
    pending_color_since: float = 0.0
    last_stats_key: object | None = None
    last_stats_time: float = 0.0
    runtime_stats: dict[str, object] = field(default_factory=dict)


_CACHE: dict[tuple[str, int], GauSplaData] = {}
_RAW_CACHE: dict[str, PlyRawData] = {}


def clear():
    _CACHE.clear()
    _RAW_CACHE.clear()


def canonical_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def _key(path: str, max_points: int) -> tuple[str, int]:
    return (canonical_path(path), int(max_points) if max_points and max_points > 0 else 0)


def get(path: str, *, max_points: int = 0) -> GauSplaData | None:
    return _CACHE.get(_key(path, max_points))


def get_raw(path: str) -> PlyRawData | None:
    return _RAW_CACHE.get(canonical_path(path))


def invalidate_batches(data: GauSplaData) -> None:
    data.gpu_batch = None
    data.gpu_batch_oit = None
    data.gpu_batch_sorted = None
    data.gpu_batch_refplus = None
    data.gpu_sort_key = None
    data.gpu_refplus_key = None


def _build_data(cpath: str, ply_data: GaussianPlyData) -> GauSplaData:
    data = GauSplaData(
        path=cpath,
        point_count=len(ply_data.positions),
        positions=ply_data.positions,
        colors=ply_data.colors,
        opacity_raw=ply_data.opacity_raw,
        opacity_draw=ply_data.opacity_draw,
        scale_raw=ply_data.scale_raw,
        rot_raw=ply_data.rot_raw,
        rot_draw=ply_data.rot_draw,
        sh_coeffs=ply_data.sh_coeffs,
        sh_degree=int(ply_data.sh_degree),
        color_source=str(ply_data.color_source),
        loaded_at=time.time(),
        error=None,
        scale_log_upper=_estimate_log_scale_upper(ply_data.scale_raw),
        filtered_count=0,
        gpu_batch=None,
        gpu_batch_oit=None,
        gpu_batch_sorted=None,
        gpu_batch_refplus=None,
        gpu_sort_key=None,
        gpu_sort_last_time=0.0,
        gpu_refplus_key=None,
        gpu_refplus_last_time=0.0,
        last_color_key=None,
        pending_color_key=None,
        pending_color_since=0.0,
        last_stats_key=None,
        last_stats_time=0.0,
        runtime_stats={},
    )
    _apply_outlier_filter(data)
    return data


def load(path: str, *, max_points: int = 0, force: bool = False) -> GauSplaData:
    cpath = canonical_path(path)
    cache_key = _key(cpath, max_points)
    if not force and cache_key in _CACHE and _CACHE[cache_key].error is None:
        return _CACHE[cache_key]

    try:
        ply_data = load_gaussian_ply(cpath, max_points=max_points)
        data = _build_data(cpath, ply_data)
    except (OSError, PlyError) as e:
        data = GauSplaData(
            path=cpath,
            point_count=0,
            positions=[],
            colors=[],
            opacity_raw=[],
            opacity_draw=[],
            scale_raw=[],
            rot_raw=[],
            rot_draw=[],
            sh_coeffs=None,
            sh_degree=0,
            color_source="RGB",
            loaded_at=time.time(),
            error=str(e),
        )

    _CACHE[cache_key] = data
    return data


def reload_if_valid(path: str, *, max_points: int = 0) -> GauSplaData:
    cpath = canonical_path(path)
    cache_key = _key(cpath, max_points)
    try:
        ply_data = load_gaussian_ply(cpath, max_points=max_points)
        data = _build_data(cpath, ply_data)
    except (OSError, PlyError) as e:
        raise RuntimeError(str(e)) from e

    for key in tuple(_CACHE.keys()):
        if key[0] == cpath and key != cache_key:
            _CACHE.pop(key, None)
    _CACHE[cache_key] = data
    _RAW_CACHE.pop(cpath, None)
    return data


def load_raw(path: str, *, force: bool = False) -> PlyRawData:
    cpath = canonical_path(path)
    if not force and cpath in _RAW_CACHE:
        return _RAW_CACHE[cpath]
    raw = load_gaussian_ply_raw(cpath)
    _RAW_CACHE[cpath] = raw
    return raw


def _estimate_log_scale_upper(scale_raw) -> float | None:
    try:
        import numpy as np  # type: ignore

        if not hasattr(scale_raw, "shape") or int(scale_raw.shape[0]) <= 0:
            return None

        sample = scale_raw
        if int(sample.shape[0]) > 200_000:
            sample = sample[:: int(sample.shape[0] / 200_000) + 1]

        neg_ratio = float(np.mean(sample < 0.0))
        if neg_ratio < 0.95:
            return None

        per_point_max = np.max(sample, axis=1)
        q999 = float(np.quantile(per_point_max, 0.999))
        if not np.isfinite(q999) or q999 < -0.25:
            return None

        return float(min(0.0, q999 + 0.15))
    except Exception:
        return None


def _apply_outlier_filter(data: GauSplaData) -> None:
    try:
        import numpy as np  # type: ignore

        if not hasattr(data.opacity_raw, "shape") or int(data.opacity_raw.shape[0]) == 0:
            return

        opacity_draw = data.opacity_raw.astype(np.float32, copy=True)
        scale = data.scale_raw.astype(np.float32, copy=False)
        quat = data.rot_draw.astype(np.float32, copy=False)

        if np.any(~np.isfinite(scale)):
            bad_scale = ~np.isfinite(scale).all(axis=1)
        else:
            bad_scale = np.zeros((scale.shape[0],), dtype=bool)
        quat_norm = np.linalg.norm(quat, axis=1)
        bad_quat = (~np.isfinite(quat).all(axis=1)) | (quat_norm < 1e-8)

        alpha = opacity_draw
        if np.min(alpha) < 0.0 or np.max(alpha) > 1.0:
            clipped = np.clip(alpha, -14.0, 14.0)
            alpha = 1.0 / (1.0 + np.exp(-clipped))

        scale_max = np.max(scale, axis=1)
        outlier = bad_scale | bad_quat
        if data.scale_log_upper is not None:
            outlier |= (scale_max > (float(data.scale_log_upper) + 0.12)) & (alpha < 0.28)
        else:
            q999 = float(np.quantile(scale_max, 0.999))
            outlier |= (scale_max > q999 + 0.15) & (alpha < 0.22)

        if np.any(outlier):
            opacity_draw[outlier] = -40.0

        data.opacity_draw = opacity_draw.astype(np.float32, copy=False)
        data.filtered_count = int(np.count_nonzero(outlier))
    except Exception:
        data.filtered_count = 0


def ensure_view_colors(data: GauSplaData, *, model_matrix, view_matrix, color_key, idle_delay: float = 0.22) -> None:
    if data.sh_coeffs is None or data.sh_degree <= 0 or data.error:
        return
    if data.last_color_key == color_key:
        data.pending_color_key = color_key
        return

    now = time.perf_counter()
    if data.pending_color_key != color_key:
        data.pending_color_key = color_key
        data.pending_color_since = now
        return
    if (now - float(data.pending_color_since)) < float(idle_delay):
        return

    try:
        import numpy as np  # type: ignore

        if not hasattr(data.positions, "shape"):
            return

        mw = np.asarray(model_matrix, dtype=np.float32).reshape((4, 4))
        vm = np.asarray(view_matrix, dtype=np.float32).reshape((4, 4))
        cam_world = np.linalg.inv(vm)[:3, 3]
        pos = data.positions.astype(np.float32, copy=False)
        world = pos @ mw[:3, :3].T + mw[:3, 3]
        dirs = cam_world[None, :] - world
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / np.maximum(norms, 1e-8)

        data.colors = eval_sh_rgb(data.sh_coeffs, dirs, int(data.sh_degree))
        data.last_color_key = color_key
        data.pending_color_key = color_key
        data.pending_color_since = now
        invalidate_batches(data)
    except Exception:
        # Keep DC preview if SH evaluation fails.
        return


def update_runtime_stats(
    data: GauSplaData,
    *,
    model_matrix,
    view_matrix,
    projection_matrix,
    viewport_size: tuple[float, float],
    scale_is_log: bool,
    size_multiplier: float,
    scene_scale: float,
    sigma_clip: float,
    max_point_size: float,
    stats_key,
) -> dict[str, object]:
    if data.last_stats_key == stats_key and data.runtime_stats:
        return data.runtime_stats

    stats = {
        "clamp_ratio": 0.0,
        "tiny_ratio": 0.0,
        "filtered_count": int(data.filtered_count),
    }
    now = time.perf_counter()
    if data.runtime_stats and (now - float(data.last_stats_time)) < 0.45:
        return data.runtime_stats

    try:
        import numpy as np  # type: ignore

        if not hasattr(data.positions, "shape") or int(data.positions.shape[0]) == 0:
            data.runtime_stats = stats
            data.last_stats_key = stats_key
            return stats

        pos = data.positions
        scale = data.scale_raw
        sample_step = 1
        if int(pos.shape[0]) > 120_000:
            sample_step = int(pos.shape[0] / 120_000) + 1
            pos = pos[::sample_step]
            scale = scale[::sample_step]

        mw = np.asarray(model_matrix, dtype=np.float32).reshape((4, 4))
        vm = np.asarray(view_matrix, dtype=np.float32).reshape((4, 4))
        pm = np.asarray(projection_matrix, dtype=np.float32).reshape((4, 4))

        view = (pos @ mw[:3, :3].T + mw[:3, 3]) @ vm[:3, :3].T + vm[:3, 3]
        fx = float(pm[0, 0]) * float(viewport_size[0]) * 0.5
        fy = float(pm[1, 1]) * float(viewport_size[1]) * 0.5
        z = np.maximum(-view[:, 2], 1e-4)

        if scale_is_log:
            scale_eval = np.exp(np.minimum(scale.astype(np.float32, copy=False), float(data.scale_log_upper or 999.0)))
        else:
            scale_eval = scale.astype(np.float32, copy=False)
        smax = np.max(scale_eval, axis=1) * float(size_multiplier) * float(scene_scale)
        radius_px = float(sigma_clip) * np.maximum(fx, fy) * (smax / z)
        point_px = 2.0 * radius_px
        stats["clamp_ratio"] = float(np.mean(point_px > float(max_point_size)))
        stats["tiny_ratio"] = float(np.mean(point_px < 1.25))
    except Exception:
        pass

    data.runtime_stats = stats
    data.last_stats_key = stats_key
    data.last_stats_time = now
    return stats
