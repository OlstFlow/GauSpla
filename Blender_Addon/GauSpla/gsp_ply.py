from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterable

from .gsp_utils import SH_C0, logit, srgb_to_linear


_PLY_TYPE_TO_STRUCT = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


@dataclass(frozen=True)
class PlyHeader:
    fmt: str
    vertex_count: int
    vertex_props: list[tuple[str, str]]


@dataclass
class GaussianPlyData:
    header: PlyHeader
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


@dataclass
class PlyRawData:
    header: PlyHeader
    rows: object


class PlyError(RuntimeError):
    pass


def _readline_ascii(f: BinaryIO) -> str:
    line = f.readline()
    if not line:
        raise PlyError("Unexpected EOF while reading header")
    return line.decode("utf-8", errors="replace").rstrip("\r\n")


def _parse_header(f: BinaryIO) -> PlyHeader:
    first = _readline_ascii(f)
    if first.strip() != "ply":
        raise PlyError("Not a PLY file (missing 'ply' magic)")

    fmt: str | None = None
    vertex_count: int | None = None
    in_vertex = False
    vertex_props: list[tuple[str, str]] = []

    while True:
        line = _readline_ascii(f).strip()
        if line == "end_header":
            break
        if not line or line.startswith("comment"):
            continue

        parts = line.split()
        if parts[0] == "format":
            if len(parts) < 3:
                raise PlyError(f"Invalid format line: {line!r}")
            fmt = parts[1]
            continue

        if parts[0] == "element":
            if len(parts) != 3:
                raise PlyError(f"Invalid element line: {line!r}")
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                try:
                    vertex_count = int(parts[2])
                except ValueError as e:
                    raise PlyError(f"Invalid vertex count: {parts[2]!r}") from e
            continue

        if parts[0] == "property" and in_vertex:
            if len(parts) >= 3 and parts[1] == "list":
                raise PlyError("Vertex list properties are not supported")
            if len(parts) != 3:
                raise PlyError(f"Invalid property line: {line!r}")
            vertex_props.append((parts[2], parts[1].lower()))
            continue

    if fmt is None:
        raise PlyError("Missing 'format' in PLY header")
    if vertex_count is None:
        raise PlyError("Missing 'element vertex' in PLY header")
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise PlyError(f"Unsupported PLY format: {fmt!r}")

    return PlyHeader(fmt=fmt, vertex_count=vertex_count, vertex_props=vertex_props)


def _indices(props: list[tuple[str, str]]) -> dict[str, int]:
    return {name: i for i, (name, _t) in enumerate(props)}


def _numpy_dtype_for_prop(np, ply_type: str, endian: str):
    if ply_type in ("float", "float32"):
        return np.dtype(endian + "f4")
    if ply_type in ("double", "float64"):
        return np.dtype(endian + "f8")
    if ply_type in ("int", "int32"):
        return np.dtype(endian + "i4")
    if ply_type in ("uint", "uint32"):
        return np.dtype(endian + "u4")
    if ply_type in ("short", "int16"):
        return np.dtype(endian + "i2")
    if ply_type in ("ushort", "uint16"):
        return np.dtype(endian + "u2")
    if ply_type in ("char", "int8"):
        return np.dtype("i1")
    if ply_type in ("uchar", "uint8"):
        return np.dtype("u1")
    raise PlyError(f"Unsupported PLY property type: {ply_type!r}")


def _numpy_dtype_for_props(np, props: list[tuple[str, str]], endian: str):
    return np.dtype([(name, _numpy_dtype_for_prop(np, ply_type, endian)) for name, ply_type in props])


def _empty_numpy_draw_state(np, count: int):
    opacity = np.full((count,), logit(1.0), dtype=np.float32)
    rot = np.zeros((count, 4), dtype=np.float32)
    rot[:, 0] = 1.0
    return opacity, rot


def _infer_sh_degree(total_coeffs: int) -> int:
    degree = int(round(math.sqrt(float(total_coeffs)) - 1.0))
    if degree < 0:
        return 0
    if (degree + 1) * (degree + 1) != total_coeffs:
        for fallback in (3, 2, 1, 0):
            if (fallback + 1) * (fallback + 1) <= total_coeffs:
                return fallback
        return 0
    return degree


def _extract_sh_coeffs_np(arr):
    names = set(arr.dtype.names or ())
    if not {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        return None, 0, "RGB"

    import numpy as np  # type: ignore

    rest_names = sorted(
        [name for name in names if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )

    total_coeffs = 1 + (len(rest_names) // 3)
    degree = _infer_sh_degree(total_coeffs)
    coeff_count = min((degree + 1) * (degree + 1), total_coeffs)
    if coeff_count <= 0:
        return None, 0, "DC"

    coeffs = np.zeros((arr.shape[0], coeff_count, 3), dtype=np.float32)
    coeffs[:, 0, 0] = arr["f_dc_0"].astype(np.float32, copy=False)
    coeffs[:, 0, 1] = arr["f_dc_1"].astype(np.float32, copy=False)
    coeffs[:, 0, 2] = arr["f_dc_2"].astype(np.float32, copy=False)

    rest_coeffs = coeff_count - 1
    if rest_coeffs > 0 and len(rest_names) >= rest_coeffs * 3:
        flat = np.stack([arr[name].astype(np.float32, copy=False) for name in rest_names[: rest_coeffs * 3]], axis=1)
        flat = flat.reshape((arr.shape[0], 3, rest_coeffs)).transpose((0, 2, 1))
        coeffs[:, 1 : 1 + rest_coeffs, :] = flat

    color_source = "SH" if degree > 0 else "DC"
    return coeffs, degree, color_source


def _extract_rgb_np(arr):
    names = set(arr.dtype.names or ())
    import numpy as np  # type: ignore

    if {"red", "green", "blue"} <= names:
        r = arr["red"].astype(np.float32, copy=False)
        g = arr["green"].astype(np.float32, copy=False)
        b = arr["blue"].astype(np.float32, copy=False)
        if arr["red"].dtype.kind in ("i", "u"):
            div = 255.0
        else:
            div = 255.0 if float(np.max(np.maximum(np.maximum(r, g), b))) > 1.0 else 1.0
        rgb = np.stack((r / div, g / div, b / div), axis=1).clip(0.0, 1.0).astype(np.float32, copy=False)
        rgb = np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            np.power((rgb + 0.055) / 1.055, 2.4),
        ).astype(np.float32, copy=False)
        return rgb, "RGB"

    coeffs, degree, source = _extract_sh_coeffs_np(arr)
    if coeffs is not None:
        col = np.clip(0.5 + SH_C0 * coeffs[:, 0, :], 0.0, 1.0).astype(np.float32, copy=False)
        col = np.where(
            col <= 0.04045,
            col / 12.92,
            np.power((col + 0.055) / 1.055, 2.4),
        ).astype(np.float32, copy=False)
        return col, source

    ones = np.ones((arr.shape[0], 3), dtype=np.float32)
    return ones, "RGB"


def _extract_opacity_raw_np(arr):
    names = set(arr.dtype.names or ())
    import numpy as np  # type: ignore

    if "opacity" in names:
        return arr["opacity"].astype(np.float32, copy=False)
    if "alpha" in names:
        a = arr["alpha"].astype(np.float32, copy=False)
        a = np.where(a > 1.0, a / 255.0, a)
        a = np.clip(a, 1e-6, 1.0 - 1e-6)
        return np.log(a / (1.0 - a)).astype(np.float32, copy=False)
    return np.full((arr.shape[0],), logit(1.0), dtype=np.float32)


def _extract_scale_raw_np(arr):
    names = set(arr.dtype.names or ())
    import numpy as np  # type: ignore

    if {"scale_0", "scale_1", "scale_2"} <= names:
        return np.stack(
            (
                arr["scale_0"].astype(np.float32, copy=False),
                arr["scale_1"].astype(np.float32, copy=False),
                arr["scale_2"].astype(np.float32, copy=False),
            ),
            axis=1,
        )
    for fallback_name in ("scale", "radius", "point_size"):
        if fallback_name in names:
            s = arr[fallback_name].astype(np.float32, copy=False)
            return np.stack((s, s, s), axis=1)
    return np.full((arr.shape[0], 3), -5.0, dtype=np.float32)


def _extract_quat_raw_np(arr):
    names = set(arr.dtype.names or ())
    import numpy as np  # type: ignore

    out = np.zeros((arr.shape[0], 4), dtype=np.float32)
    out[:, 0] = 1.0
    if {"rot_0", "rot_1", "rot_2", "rot_3"} <= names:
        out[:, 0] = arr["rot_0"].astype(np.float32, copy=False)
        out[:, 1] = arr["rot_1"].astype(np.float32, copy=False)
        out[:, 2] = arr["rot_2"].astype(np.float32, copy=False)
        out[:, 3] = arr["rot_3"].astype(np.float32, copy=False)
    return out


def _sanitize_quat_np(quat):
    import numpy as np  # type: ignore

    out = quat.astype(np.float32, copy=True)
    bad = ~np.isfinite(out).all(axis=1)
    norms = np.linalg.norm(out, axis=1)
    bad |= norms < 1e-8
    if np.any(bad):
        out[bad, :] = (1.0, 0.0, 0.0, 0.0)
    return out


def _cast_ascii_value(value: str, ply_type: str):
    if ply_type in ("char", "int8", "uchar", "uint8", "short", "int16", "ushort", "uint16", "int", "int32", "uint", "uint32"):
        return int(value)
    return float(value)


def _extract_rgb(row: Iterable, prop_i: dict[str, int], prop_t: dict[str, str] | None = None) -> tuple[float, float, float]:
    if {"red", "green", "blue"} <= prop_i.keys():
        r0 = float(row[prop_i["red"]])
        g0 = float(row[prop_i["green"]])
        b0 = float(row[prop_i["blue"]])
        div = 1.0
        if prop_t is not None and (prop_t.get("red") or "").lower() in ("uchar", "uint8"):
            div = 255.0
        if div == 1.0 and max(r0, g0, b0) > 1.0:
            div = 255.0
        return (
            srgb_to_linear(r0 / div),
            srgb_to_linear(g0 / div),
            srgb_to_linear(b0 / div),
        )

    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= prop_i.keys():
        return (
            srgb_to_linear(0.5 + SH_C0 * float(row[prop_i["f_dc_0"]])),
            srgb_to_linear(0.5 + SH_C0 * float(row[prop_i["f_dc_1"]])),
            srgb_to_linear(0.5 + SH_C0 * float(row[prop_i["f_dc_2"]])),
        )

    return (1.0, 1.0, 1.0)


def _extract_opacity_raw(row: Iterable, prop_i: dict[str, int]) -> float:
    if "opacity" in prop_i:
        return float(row[prop_i["opacity"]])
    if "alpha" in prop_i:
        a = float(row[prop_i["alpha"]])
        if a > 1.0:
            a /= 255.0
        a = max(1e-6, min(1.0 - 1e-6, a))
        return logit(a)
    return logit(1.0)


def _extract_scale_raw(row: Iterable, prop_i: dict[str, int]) -> tuple[float, float, float]:
    if {"scale_0", "scale_1", "scale_2"} <= prop_i.keys():
        return (
            float(row[prop_i["scale_0"]]),
            float(row[prop_i["scale_1"]]),
            float(row[prop_i["scale_2"]]),
        )
    for fallback_name in ("scale", "radius", "point_size"):
        if fallback_name in prop_i:
            s = float(row[prop_i[fallback_name]])
            return (s, s, s)
    return (-5.0, -5.0, -5.0)


def _extract_quat_raw(row: Iterable, prop_i: dict[str, int]) -> tuple[float, float, float, float]:
    if {"rot_0", "rot_1", "rot_2", "rot_3"} <= prop_i.keys():
        return (
            float(row[prop_i["rot_0"]]),
            float(row[prop_i["rot_1"]]),
            float(row[prop_i["rot_2"]]),
            float(row[prop_i["rot_3"]]),
        )
    return (1.0, 0.0, 0.0, 0.0)


def _load_numpy_preview(arr, header: PlyHeader) -> GaussianPlyData:
    import numpy as np  # type: ignore

    positions = np.stack(
        (
            arr["x"].astype(np.float32, copy=False),
            arr["y"].astype(np.float32, copy=False),
            arr["z"].astype(np.float32, copy=False),
        ),
        axis=1,
    )
    colors, color_source = _extract_rgb_np(arr)
    opacity_raw = _extract_opacity_raw_np(arr)
    opacity_draw = opacity_raw.astype(np.float32, copy=True)
    scale_raw = _extract_scale_raw_np(arr)
    rot_raw = _extract_quat_raw_np(arr)
    rot_draw = _sanitize_quat_np(rot_raw)
    sh_coeffs, sh_degree, sh_source = _extract_sh_coeffs_np(arr)
    if sh_coeffs is not None:
        color_source = sh_source

    return GaussianPlyData(
        header=header,
        positions=positions.astype(np.float32, copy=False),
        colors=colors.astype(np.float32, copy=False),
        opacity_raw=opacity_raw.astype(np.float32, copy=False),
        opacity_draw=opacity_draw,
        scale_raw=scale_raw.astype(np.float32, copy=False),
        rot_raw=rot_raw.astype(np.float32, copy=False),
        rot_draw=rot_draw.astype(np.float32, copy=False),
        sh_coeffs=sh_coeffs.astype(np.float32, copy=False) if sh_coeffs is not None else None,
        sh_degree=int(sh_degree),
        color_source=str(color_source),
    )


def _load_ascii_preview(f: BinaryIO, header: PlyHeader, max_points: int) -> GaussianPlyData:
    props = header.vertex_props
    idx = _indices(props)
    prop_t = {name: ply_type for name, ply_type in props}
    if not {"x", "y", "z"} <= set(idx.keys()):
        raise PlyError("PLY is missing x/y/z vertex properties")

    n = header.vertex_count
    stride = 1
    if max_points and max_points > 0 and n > max_points:
        stride = max(1, int(math.ceil(n / max_points)))

    pos = []
    colors = []
    opacity_raw = []
    scale_raw = []
    rot_raw = []

    for row_i in range(n):
        line = f.readline()
        if not line:
            raise PlyError("Unexpected EOF while reading vertex data")
        if row_i % stride != 0:
            continue
        parts = line.decode("utf-8", errors="replace").split()
        if len(parts) < len(props):
            raise PlyError("Vertex line has fewer fields than header properties")
        values = [_cast_ascii_value(parts[i], props[i][1]) for i in range(len(props))]
        pos.append((float(values[idx["x"]]), float(values[idx["y"]]), float(values[idx["z"]])))
        colors.append(_extract_rgb(values, idx, prop_t))
        opacity_raw.append(_extract_opacity_raw(values, idx))
        scale_raw.append(_extract_scale_raw(values, idx))
        rot_raw.append(_extract_quat_raw(values, idx))

    try:
        import numpy as np  # type: ignore

        pos_np = np.asarray(pos, dtype=np.float32)
        colors_np = np.asarray(colors, dtype=np.float32)
        opacity_np = np.asarray(opacity_raw, dtype=np.float32)
        scale_np = np.asarray(scale_raw, dtype=np.float32)
        rot_np = np.asarray(rot_raw, dtype=np.float32)
        return GaussianPlyData(
            header=header,
            positions=pos_np,
            colors=colors_np,
            opacity_raw=opacity_np,
            opacity_draw=opacity_np.copy(),
            scale_raw=scale_np,
            rot_raw=rot_np,
            rot_draw=_sanitize_quat_np(rot_np),
            sh_coeffs=None,
            sh_degree=0,
            color_source="DC" if {"f_dc_0", "f_dc_1", "f_dc_2"} <= set(idx.keys()) else "RGB",
        )
    except Exception:
        return GaussianPlyData(
            header=header,
            positions=pos,
            colors=colors,
            opacity_raw=opacity_raw,
            opacity_draw=list(opacity_raw),
            scale_raw=scale_raw,
            rot_raw=rot_raw,
            rot_draw=list(rot_raw),
            sh_coeffs=None,
            sh_degree=0,
            color_source="DC" if {"f_dc_0", "f_dc_1", "f_dc_2"} <= set(idx.keys()) else "RGB",
        )


def _load_binary_preview(f: BinaryIO, header: PlyHeader, max_points: int, *, endian: str) -> GaussianPlyData:
    n = header.vertex_count
    row_fmt = endian + "".join(_PLY_TYPE_TO_STRUCT.get(ply_type, "") for _name, ply_type in header.vertex_props)
    if not row_fmt or any(_PLY_TYPE_TO_STRUCT.get(ply_type) is None for _name, ply_type in header.vertex_props):
        bad = [ply_type for _name, ply_type in header.vertex_props if _PLY_TYPE_TO_STRUCT.get(ply_type) is None]
        raise PlyError(f"Unsupported PLY property type: {bad[0]!r}")

    row_size = struct.calcsize(row_fmt)
    blob = f.read(row_size * n)
    if len(blob) != row_size * n:
        raise PlyError("Unexpected EOF while reading binary vertex data")

    try:
        import numpy as np  # type: ignore

        dtype = _numpy_dtype_for_props(np, header.vertex_props, endian)
        arr = np.frombuffer(blob, dtype=dtype, count=n)
        if max_points and max_points > 0 and n > max_points:
            stride = max(1, int(math.ceil(n / max_points)))
            arr = arr[::stride]
        return _load_numpy_preview(arr, header)
    except Exception:
        props = header.vertex_props
        idx = _indices(props)
        prop_t = {name: ply_type for name, ply_type in props}
        pos = []
        colors = []
        opacity_raw = []
        scale_raw = []
        rot_raw = []
        stride = 1
        if max_points and max_points > 0 and n > max_points:
            stride = max(1, int(math.ceil(n / max_points)))
        for row_i, row in enumerate(struct.iter_unpack(row_fmt, blob)):
            if row_i % stride != 0:
                continue
            pos.append((float(row[idx["x"]]), float(row[idx["y"]]), float(row[idx["z"]])))
            colors.append(_extract_rgb(row, idx, prop_t))
            opacity_raw.append(_extract_opacity_raw(row, idx))
            scale_raw.append(_extract_scale_raw(row, idx))
            rot_raw.append(_extract_quat_raw(row, idx))
        return GaussianPlyData(
            header=header,
            positions=pos,
            colors=colors,
            opacity_raw=opacity_raw,
            opacity_draw=list(opacity_raw),
            scale_raw=scale_raw,
            rot_raw=rot_raw,
            rot_draw=list(rot_raw),
            sh_coeffs=None,
            sh_degree=0,
            color_source="RGB",
        )


def load_gaussian_ply(path: str, *, max_points: int = 0) -> GaussianPlyData:
    if not os.path.exists(path):
        raise PlyError(f"File not found: {path}")

    with open(path, "rb") as f:
        header = _parse_header(f)
        if header.fmt == "ascii":
            return _load_ascii_preview(f, header, max_points)
        if header.fmt == "binary_little_endian":
            return _load_binary_preview(f, header, max_points, endian="<")
        return _load_binary_preview(f, header, max_points, endian=">")


def load_gaussian_ply_raw(path: str) -> PlyRawData:
    if not os.path.exists(path):
        raise PlyError(f"File not found: {path}")

    with open(path, "rb") as f:
        header = _parse_header(f)
        if header.fmt == "ascii":
            props = header.vertex_props
            rows = []
            for _row_i in range(header.vertex_count):
                line = f.readline()
                if not line:
                    raise PlyError("Unexpected EOF while reading vertex data")
                parts = line.decode("utf-8", errors="replace").split()
                if len(parts) < len(props):
                    raise PlyError("Vertex line has fewer fields than header properties")
                rows.append(tuple(_cast_ascii_value(parts[i], props[i][1]) for i in range(len(props))))
            return PlyRawData(header=header, rows=rows)

        endian = "<" if header.fmt == "binary_little_endian" else ">"
        row_fmt = endian + "".join(_PLY_TYPE_TO_STRUCT.get(ply_type, "") for _name, ply_type in header.vertex_props)
        if not row_fmt or any(_PLY_TYPE_TO_STRUCT.get(ply_type) is None for _name, ply_type in header.vertex_props):
            bad = [ply_type for _name, ply_type in header.vertex_props if _PLY_TYPE_TO_STRUCT.get(ply_type) is None]
            raise PlyError(f"Unsupported PLY property type: {bad[0]!r}")

        blob = f.read(struct.calcsize(row_fmt) * header.vertex_count)
        if len(blob) != struct.calcsize(row_fmt) * header.vertex_count:
            raise PlyError("Unexpected EOF while reading binary vertex data")

        try:
            import numpy as np  # type: ignore

            dtype = _numpy_dtype_for_props(np, header.vertex_props, endian)
            rows = np.frombuffer(blob, dtype=dtype, count=header.vertex_count).copy()
            return PlyRawData(header=header, rows=rows)
        except Exception:
            rows = list(struct.iter_unpack(row_fmt, blob))
            return PlyRawData(header=header, rows=rows)


def write_gaussian_ply_subset(path: str, raw_data: PlyRawData, keep_indices) -> None:
    props = raw_data.header.vertex_props
    keep_count = len(keep_indices)
    with open(path, "wb") as f:
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {keep_count}",
        ]
        for name, ply_type in props:
            header_lines.append(f"property {ply_type} {name}")
        header_lines.append("end_header")
        f.write(("\n".join(header_lines) + "\n").encode("ascii"))

        rows = raw_data.rows
        try:
            import numpy as np  # type: ignore

            if hasattr(rows, "dtype") and hasattr(rows, "shape"):
                index_arr = np.asarray(keep_indices, dtype=np.int64)
                subset = rows[index_arr]
                subset = subset.astype(rows.dtype.newbyteorder("<"), copy=False)
                f.write(subset.tobytes())
                return
        except Exception:
            pass

        row_fmt = "<" + "".join(_PLY_TYPE_TO_STRUCT[ply_type] for _name, ply_type in props)
        pack = struct.Struct(row_fmt).pack
        for idx in keep_indices:
            f.write(pack(*rows[int(idx)]))
