from __future__ import annotations

import math


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def srgb_to_linear(value: float) -> float:
    value = clamp01(value)
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float) -> float:
    p = clamp01(p)
    # avoid inf
    if p <= 1e-6:
        p = 1e-6
    elif p >= 1.0 - 1e-6:
        p = 1.0 - 1e-6
    return math.log(p / (1.0 - p))


def srgb_from_f_dc(f_dc0: float, f_dc1: float, f_dc2: float) -> tuple[float, float, float]:
    r = srgb_to_linear(0.5 + SH_C0 * f_dc0)
    g = srgb_to_linear(0.5 + SH_C0 * f_dc1)
    b = srgb_to_linear(0.5 + SH_C0 * f_dc2)
    return (r, g, b)


def _eval_sh_rgb_python(coeffs, direction: tuple[float, float, float], degree: int) -> tuple[float, float, float]:
    x, y, z = direction
    out = [SH_C0 * float(coeffs[0][c]) for c in range(3)]

    if degree >= 1 and len(coeffs) >= 4:
        out[0] += -SH_C1 * y * float(coeffs[1][0]) + SH_C1 * z * float(coeffs[2][0]) - SH_C1 * x * float(coeffs[3][0])
        out[1] += -SH_C1 * y * float(coeffs[1][1]) + SH_C1 * z * float(coeffs[2][1]) - SH_C1 * x * float(coeffs[3][1])
        out[2] += -SH_C1 * y * float(coeffs[1][2]) + SH_C1 * z * float(coeffs[2][2]) - SH_C1 * x * float(coeffs[3][2])

    if degree >= 2 and len(coeffs) >= 9:
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        yz = y * z
        xz = x * z
        basis = (
            SH_C2[0] * xy,
            SH_C2[1] * yz,
            SH_C2[2] * (2.0 * zz - xx - yy),
            SH_C2[3] * xz,
            SH_C2[4] * (xx - yy),
        )
        for i in range(5):
            c = coeffs[4 + i]
            out[0] += basis[i] * float(c[0])
            out[1] += basis[i] * float(c[1])
            out[2] += basis[i] * float(c[2])

    if degree >= 3 and len(coeffs) >= 16:
        xx = x * x
        yy = y * y
        zz = z * z
        basis = (
            SH_C3[0] * y * (3.0 * xx - yy),
            SH_C3[1] * x * y * z,
            SH_C3[2] * y * (4.0 * zz - xx - yy),
            SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            SH_C3[4] * x * (4.0 * zz - xx - yy),
            SH_C3[5] * z * (xx - yy),
            SH_C3[6] * x * (xx - 3.0 * yy),
        )
        for i in range(7):
            c = coeffs[9 + i]
            out[0] += basis[i] * float(c[0])
            out[1] += basis[i] * float(c[1])
            out[2] += basis[i] * float(c[2])

    return (
        srgb_to_linear(0.5 + out[0]),
        srgb_to_linear(0.5 + out[1]),
        srgb_to_linear(0.5 + out[2]),
    )


def eval_sh_rgb(coeffs, direction, degree: int):
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(coeffs, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]

        dirs = np.asarray(direction, dtype=np.float32)
        if dirs.ndim == 1:
            dirs = dirs[None, ...]

        out = SH_C0 * arr[:, 0, :]
        x = dirs[:, 0]
        y = dirs[:, 1]
        z = dirs[:, 2]

        if degree >= 1 and arr.shape[1] >= 4:
            out = (
                out
                - SH_C1 * y[:, None] * arr[:, 1, :]
                + SH_C1 * z[:, None] * arr[:, 2, :]
                - SH_C1 * x[:, None] * arr[:, 3, :]
            )

        if degree >= 2 and arr.shape[1] >= 9:
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z
            out = (
                out
                + SH_C2[0] * xy[:, None] * arr[:, 4, :]
                + SH_C2[1] * yz[:, None] * arr[:, 5, :]
                + SH_C2[2] * (2.0 * zz - xx - yy)[:, None] * arr[:, 6, :]
                + SH_C2[3] * xz[:, None] * arr[:, 7, :]
                + SH_C2[4] * (xx - yy)[:, None] * arr[:, 8, :]
            )

        if degree >= 3 and arr.shape[1] >= 16:
            xx = x * x
            yy = y * y
            zz = z * z
            out = (
                out
                + SH_C3[0] * (y * (3.0 * xx - yy))[:, None] * arr[:, 9, :]
                + SH_C3[1] * (x * y * z)[:, None] * arr[:, 10, :]
                + SH_C3[2] * (y * (4.0 * zz - xx - yy))[:, None] * arr[:, 11, :]
                + SH_C3[3] * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy))[:, None] * arr[:, 12, :]
                + SH_C3[4] * (x * (4.0 * zz - xx - yy))[:, None] * arr[:, 13, :]
                + SH_C3[5] * (z * (xx - yy))[:, None] * arr[:, 14, :]
                + SH_C3[6] * (x * (xx - 3.0 * yy))[:, None] * arr[:, 15, :]
            )

        out = np.clip(0.5 + out, 0.0, 1.0).astype(np.float32, copy=False)
        out = np.where(
            out <= 0.04045,
            out / 12.92,
            np.power((out + 0.055) / 1.055, 2.4),
        ).astype(np.float32, copy=False)
        if np.asarray(coeffs).ndim == 2:
            return out[0]
        return out
    except Exception:
        return _eval_sh_rgb_python(coeffs, tuple(direction), degree)
