from __future__ import annotations

import time

import bpy

from . import gsp_cache
from .gsp_props import OBJECT_SETTINGS_ATTR, color_match_label, ensure_look_preset_initialized


_DRAW_HANDLER = None
_SHADER = None
_SHADER_REFERENCE_PLUS = None
_REFERENCE_PLUS_TILE_PX = 10.0
_SCENE_DEPTH_TEX = None
_SCENE_DEPTH_SIZE = (0, 0)
_DUMMY_DEPTH_TEX = None
_PARAMS_UBO = None
_PERF_LAST = 0.0
_PERF_DT_EMA = 0.0
_PERF_FPS = 0.0
_LOD_STATE: dict[int, dict[str, float]] = {}
_LAST_DRAW_ERROR: str | None = None

_PARAMS_STRUCT_SOURCE = """
struct GauSplaParams {
    mat4 view_matrix;
    mat4 projection_matrix;
    mat4 model_matrix;
    vec4 viewport_persp;
    vec4 render_params;
    vec4 view_params;
    vec4 shape_params;
    vec4 alpha_params;
    vec4 debug_params;
    vec4 color_params;
    vec4 extra_params;
};
"""

_PARAMS_DEFINE_MAP = {
    "u_ViewMatrix": "u_Params.view_matrix",
    "u_ProjectionMatrix": "u_Params.projection_matrix",
    "u_ModelMatrix": "u_Params.model_matrix",
    "u_ViewportSize": "u_Params.viewport_persp.xy",
    "u_IsPerspective": "u_Params.viewport_persp.z",
    "u_ViewDistance": "u_Params.viewport_persp.w",
    "u_DrawStride": "u_Params.render_params.x",
    "u_SceneScale": "u_Params.render_params.y",
    "u_PointMode": "u_Params.render_params.z",
    "u_PointSizePx": "u_Params.render_params.w",
    "u_StableScreenScale": "u_Params.view_params.x",
    "u_StableFovRad": "u_Params.view_params.y",
    "u_OrthoDistanceScale": "u_Params.view_params.z",
    "u_SigmaClip": "u_Params.view_params.w",
    "u_MinPixelCov": "u_Params.shape_params.x",
    "u_ScaleMultiplier": "u_Params.shape_params.y",
    "u_ScaleIsLog": "u_Params.shape_params.z",
    "u_MaxPointSize": "u_Params.shape_params.w",
    "u_AllowDistortion": "u_Params.alpha_params.x",
    "u_AlphaMultiplier": "u_Params.alpha_params.y",
    "u_QuatOrderXYZW": "u_Params.alpha_params.z",
    "u_OpacityIsLinear": "u_Params.alpha_params.w",
    "u_DebugMode": "u_Params.debug_params.x",
    "u_Isolation": "u_Params.debug_params.y",
    "u_AlphaCurve": "u_Params.debug_params.z",
    "u_ColorGain": "u_Params.debug_params.w",
    "u_ColorSaturation": "u_Params.color_params.x",
    "u_ColorGamma": "u_Params.color_params.y",
    "u_UseSceneDepth": "u_Params.color_params.z",
    "u_SceneDepthBias": "u_Params.color_params.w",
    "u_OitPass": "u_Params.extra_params.x",
    "u_LogScaleUpper": "u_Params.extra_params.y",
    "u_ColorMatch": "u_Params.extra_params.z",
}


def get_fps_estimate() -> float:
    return float(_PERF_FPS)


def get_last_draw_error() -> str | None:
    return _LAST_DRAW_ERROR


def _set_last_error(msg: str | None):
    global _LAST_DRAW_ERROR
    _LAST_DRAW_ERROR = msg


def get_runtime_lod(obj) -> tuple[int, int] | None:
    st = _LOD_STATE.get(int(obj.as_pointer()))
    if not st:
        return None
    stride = int(st.get("stride", 0.0))
    mps = int(st.get("max_point_size", 0.0))
    if stride <= 0:
        stride = 1
    return (stride, mps)


def reset_runtime_lod(obj) -> None:
    key = int(obj.as_pointer())
    if key in _LOD_STATE:
        del _LOD_STATE[key]


def _update_perf():
    global _PERF_LAST, _PERF_DT_EMA, _PERF_FPS

    now = time.perf_counter()
    if _PERF_LAST <= 0.0:
        _PERF_LAST = now
        return
    dt = now - _PERF_LAST
    _PERF_LAST = now
    if dt <= 0.0:
        return
    # Avoid huge dt spikes (no redraw / window unfocused) poisoning EMA.
    if dt > 0.5:
        return

    alpha = 0.08
    _PERF_DT_EMA = dt if _PERF_DT_EMA <= 0.0 else (_PERF_DT_EMA * (1.0 - alpha) + dt * alpha)
    if _PERF_DT_EMA > 1e-6:
        _PERF_FPS = 1.0 / _PERF_DT_EMA


def _get_lod_state(obj) -> dict[str, float]:
    key = int(obj.as_pointer())
    st = _LOD_STATE.get(key)
    if st is None:
        st = {"stride": 1.0, "max_point_size": 0.0, "last": 0.0}
        _LOD_STATE[key] = st
    return st


def _effective_stride_and_mps(obj, s, do_sort: bool) -> tuple[int, int]:
    base_stride = max(1, int(s.draw_stride))
    base_mps = max(1, int(s.max_point_size))

    if do_sort or not s.auto_lod:
        return base_stride, base_mps

    st = _get_lod_state(obj)
    stride = max(base_stride, int(st.get("stride", float(base_stride))))

    mps = base_mps
    if s.lod_adjust_point_size:
        st_mps = int(st.get("max_point_size", 0.0))
        if st_mps <= 0:
            st_mps = base_mps
        mps = min(base_mps, max(int(s.lod_min_point_size), st_mps))

    now = time.perf_counter()
    last = float(st.get("last", 0.0))
    if now - last < float(s.lod_update_interval):
        return stride, mps

    st["last"] = now

    target = float(s.target_fps)
    hyst = float(s.lod_hysteresis)
    fps = float(_PERF_FPS)

    min_stride = max(1, int(s.lod_min_stride))
    max_stride = max(min_stride, int(s.lod_max_stride))

    if fps > 1e-3 and fps < target - hyst:
        # Increase stride smoothly to avoid "jumps" that feel glitchy.
        stride = min(max_stride, int(stride * 1.2) + 1)

        if s.lod_adjust_point_size:
            min_mps = max(1, int(s.lod_min_point_size))
            mps = max(min_mps, int(mps * 0.85))
            st["max_point_size"] = float(mps)

    elif fps > target + hyst:
        stride = max(min_stride, int(stride / 1.15))
        if s.lod_adjust_point_size:
            st["max_point_size"] = float(min(base_mps, int(mps * 1.08) + 1))

    st["stride"] = float(stride)
    return stride, mps


def _add_shader_params_ubo(create_info) -> None:
    create_info.typedef_source(_PARAMS_STRUCT_SOURCE)
    create_info.uniform_buf(0, "GauSplaParams", "u_Params")
    for name, value in _PARAMS_DEFINE_MAP.items():
        create_info.define(name, value)


def _matrix_to_ubo_data(matrix) -> list[float]:
    return [float(matrix[row][col]) for col in range(4) for row in range(4)]


def _debug_mode_value(settings) -> float:
    dbg = getattr(settings, "debug_mode", "NONE")
    if dbg == "ANISO":
        return 1.0
    if dbg == "CLAMP":
        return 2.0
    return 0.0


def _opacity_is_linear(settings, isolation: bool) -> float:
    op_mode = getattr(settings, "opacity_encoding", "AUTO")
    if isolation:
        return 1.0
    if op_mode == "LINEAR":
        return 1.0
    if op_mode == "AUTO":
        return 0.0
    return 0.0


def _color_match_value(settings) -> float:
    return 1.0 if getattr(settings, "color_match", "NEUTRAL") in {"LICHTFELD", "SOFT_CONTRAST"} else 0.0


def _build_shader_params(
    *,
    view_mat,
    proj_mat,
    model_mat,
    viewport_size,
    is_perspective: float,
    view_distance: float,
    settings,
    draw_stride: int,
    max_point_size: float,
    scale_log_upper: float | None,
    point_mode: float,
    isolation: bool,
    use_scene_depth: bool,
    oit_pass: float = 0.0,
) -> list[float]:
    params: list[float] = []
    params.extend(_matrix_to_ubo_data(view_mat))
    params.extend(_matrix_to_ubo_data(proj_mat))
    params.extend(_matrix_to_ubo_data(model_mat))
    params.extend(
        [
            float(viewport_size[0]),
            float(viewport_size[1]),
            float(is_perspective),
            float(view_distance),
        ]
    )
    params.extend(
        [
            float(draw_stride),
            float(settings.scene_scale) if settings.use_scene_scale else 1.0,
            float(point_mode),
            float(settings.disabled_point_size),
        ]
    )
    params.extend(
        [
            0.0 if isolation else (1.0 if (settings.use_view_stabilization and settings.stable_screen_scale) else 0.0),
            float(3.141592653589793) * float(settings.stable_fov_deg) / 180.0,
            0.0 if isolation else (1.0 if (settings.use_view_stabilization and settings.ortho_distance_scale) else 0.0),
            float(settings.sigma_clip),
        ]
    )
    params.extend(
        [
            0.0 if isolation else float(settings.min_pixel_cov),
            float(settings.size_multiplier),
            1.0 if (isolation or settings.scale_is_log) else 0.0,
            float(max_point_size),
        ]
    )
    params.extend(
        [
            0.0 if isolation else (1.0 if settings.allow_splat_distortion else 0.0),
            float(settings.alpha_multiplier),
            1.0 if getattr(settings, "quat_order", "WXYZ") == "XYZW" else 0.0,
            _opacity_is_linear(settings, isolation),
        ]
    )
    params.extend(
        [
            _debug_mode_value(settings),
            1.0 if isolation else 0.0,
            1.0 if isolation else float(getattr(settings, "alpha_curve", 1.0)),
            float(getattr(settings, "color_gain", 1.0)),
        ]
    )
    params.extend(
        [
            float(getattr(settings, "color_saturation", 1.0)),
            float(getattr(settings, "color_gamma", 1.0)),
            1.0 if use_scene_depth else 0.0,
            float(getattr(settings, "scene_depth_bias", 0.0002)),
        ]
    )
    params.extend(
        [
            float(oit_pass),
            float(scale_log_upper if scale_log_upper is not None else 999.0),
            _color_match_value(settings),
            0.0,
        ]
    )
    return params


def _bind_shader_params(shader, params: list[float]) -> None:
    global _PARAMS_UBO

    import gpu

    buf = gpu.types.Buffer("FLOAT", len(params), params)
    if _PARAMS_UBO is None:
        _PARAMS_UBO = gpu.types.GPUUniformBuf(buf)
    else:
        _PARAMS_UBO.update(buf)
    shader.uniform_block("u_Params", _PARAMS_UBO)


def _ensure_shader():
    global _SHADER
    if _SHADER is not None:
        return _SHADER

    import gpu

    try:
        create_info = gpu.types.GPUShaderCreateInfo()
    except Exception:
        # Fallback for older Blender builds (kept minimal).
        raise RuntimeError("GPUShaderCreateInfo is not available in this Blender build")

    _add_shader_params_ubo(create_info)

    create_info.sampler(1, "FLOAT_2D", "u_SceneDepthTex")

    create_info.vertex_in(0, "VEC3", "position")
    create_info.vertex_in(1, "VEC3", "color")
    create_info.vertex_in(2, "FLOAT", "opacity")
    create_info.vertex_in(3, "VEC3", "scale")
    create_info.vertex_in(4, "VEC4", "quat")

    iface = gpu.types.GPUStageInterfaceInfo("gau_spla_iface")
    iface.smooth("VEC3", "v_color")
    iface.smooth("FLOAT", "v_opacity_raw")
    iface.smooth("VEC3", "v_conic")  # inv00, inv01, inv11
    iface.smooth("FLOAT", "v_radius")
    iface.smooth("FLOAT", "v_alpha_preserve")
    create_info.vertex_out(iface)
    create_info.fragment_out(0, "VEC4", "fragColor")

    create_info.vertex_source(
        """
        float sigmoid(float x)
        {
            if (x >= 0.0) {
                float z = exp(-x);
                return 1.0 / (1.0 + z);
            }
            float z = exp(x);
            return z / (1.0 + z);
        }

        mat3 quatToMat3(vec4 qWXYZ)
        {
            float w = qWXYZ.x;
            float x = qWXYZ.y;
            float y = qWXYZ.z;
            float z = qWXYZ.w;

            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float xz = x * z;
            float yz = y * z;
            float wx = w * x;
            float wy = w * y;
            float wz = w * z;

            // Column-major mat3 constructor: mat3(col0, col1, col2)
            vec3 c0 = vec3(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy));
            vec3 c1 = vec3(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx));
            vec3 c2 = vec3(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy));
            return mat3(c0, c1, c2);
        }

        void main()
        {
            int stride = int(u_DrawStride + 0.5);
            if (stride > 1 && (gl_VertexID % stride) != 0) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                gl_PointSize = 0.0;
                v_color = vec3(0.0);
                v_opacity_raw = 0.0;
                v_conic = vec3(1.0, 0.0, 1.0);
                v_radius = 0.0;
                v_alpha_preserve = 1.0;
                return;
            }

            vec4 world = u_ModelMatrix * vec4(position, 1.0);
            vec4 view = u_ViewMatrix * world;
            // Cull points behind camera only in perspective (to avoid huge point sprites near z=0).
            // In Ortho we don't cull here; let the projection handle clipping.
            if (u_Isolation < 0.5 && u_IsPerspective > 0.5 && view.z >= -1e-4) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                gl_PointSize = 0.0;
                v_color = vec3(0.0);
                v_opacity_raw = 0.0;
                v_conic = vec3(1.0, 0.0, 1.0);
                v_radius = 0.0;
                v_alpha_preserve = 1.0;
                return;
            }
            gl_Position = u_ProjectionMatrix * view;

            // Point preview mode (when Enabled is off)
            if (u_PointMode > 0.5) {
                float ps = clamp(u_PointSizePx, 1.0, u_MaxPointSize);
                gl_PointSize = ps;
                v_color = color;
                v_opacity_raw = 0.0;
                v_conic = vec3(1.0, 0.0, 1.0);
                v_radius = 0.5 * ps;
                v_alpha_preserve = 1.0;
                return;
            }

            // Normalize object rotation (ignore object scale/shear for covariance orientation)
            mat3 M = mat3(u_ModelMatrix);
            vec3 objS = vec3(length(M[0]), length(M[1]), length(M[2]));
            mat3 Robj = mat3(normalize(M[0]), normalize(M[1]), normalize(M[2]));
            mat3 Rview = mat3(u_ViewMatrix);

            vec4 q = quat;
            if (u_Isolation > 0.5) {
                // Ignore quaternion in isolation mode to remove convention issues.
                q = vec4(1.0, 0.0, 0.0, 0.0);
            }
            if (u_QuatOrderXYZW > 0.5) {
                q = vec4(quat.w, quat.x, quat.y, quat.z); // XYZW -> WXYZ
            }
            q /= max(length(q), 1e-8);
            mat3 Rq = quatToMat3(q);

            float use_log = (u_Isolation > 0.5) ? 1.0 : u_ScaleIsLog;
            vec3 scale_eval = scale;
            if (use_log > 0.5 && u_LogScaleUpper < 100.0) {
                scale_eval = min(scale_eval, vec3(u_LogScaleUpper));
            }
            vec3 s = (use_log > 0.5) ? exp(scale_eval) : scale_eval;
            if (u_Isolation > 0.5) {
                // Keep it finite even for "wrong" encoding; still shows something.
                s = clamp(abs(s), vec3(1e-6), vec3(1e3));
            }
            // Apply splat-scale multipliers and object scale so scaling the empty
            // affects both point positions and splat size (more intuitive).
            s *= (u_ScaleMultiplier * u_SceneScale);
            s *= objS;

            mat3 Rcam = Rview * Robj * Rq;
            vec3 r0 = Rcam[0];
            vec3 r1 = Rcam[1];
            vec3 r2 = Rcam[2];

            float sx2 = s.x * s.x;
            float sy2 = s.y * s.y;
            float sz2 = s.z * s.z;

            // Pixel scale factors
            float fx;
            float fy;
            if (u_IsPerspective > 0.5 && u_StableScreenScale > 0.5) {
                float f = 0.5 * u_ViewportSize.y / max(tan(0.5 * u_StableFovRad), 1e-4);
                fx = f;
                fy = f;
            } else {
                fx = u_ProjectionMatrix[0][0] * u_ViewportSize.x * 0.5;
                fy = u_ProjectionMatrix[1][1] * u_ViewportSize.y * 0.5;
            }

            // Blender/OpenGL view space is typically -Z forward
            float z = max(1e-4, -view.z);
            float invz = 1.0 / z;
            float invz2 = invz * invz;

            vec3 j0;
            vec3 j1;
            if (u_IsPerspective > 0.5) {
                // Blender view space uses negative Z in front of the camera:
                // x_screen = fx * X / (-z_view), y_screen = fy * Y / (-z_view).
                j0 = vec3(fx * invz, 0.0, fx * view.x * invz2);
                j1 = vec3(0.0, fy * invz, fy * view.y * invz2);
            } else {
                // Ortho Jacobian for x = fx * X, y = fy * Y
                float zref = max(u_ViewDistance, 1e-4);
                float k = (u_OrthoDistanceScale > 0.5) ? (1.0 / zref) : 1.0;
                j0 = vec3(fx * k, 0.0, 0.0);
                j1 = vec3(0.0, fy * k, 0.0);
            }

            // Compute C = J * Sigma * J^T without building Sigma:
            // Sigma = sum_k s_k^2 * r_k r_k^T
            float a0 = dot(j0, r0);
            float a1 = dot(j0, r1);
            float a2 = dot(j0, r2);
            float b0 = dot(j1, r0);
            float b1 = dot(j1, r1);
            float b2 = dot(j1, r2);

            float c00_raw = sx2 * a0 * a0 + sy2 * a1 * a1 + sz2 * a2 * a2;
            float c01_raw = sx2 * a0 * b0 + sy2 * a1 * b1 + sz2 * a2 * b2;
            float c11_raw = sx2 * b0 * b0 + sy2 * b1 * b1 + sz2 * b2 * b2;

            float det_raw = c00_raw * c11_raw - c01_raw * c01_raw;
            float trace_raw = c00_raw + c11_raw;
            float disc_raw = max(0.0, 0.25 * trace_raw * trace_raw - det_raw);
            float sd_raw = sqrt(disc_raw);
            float lambda_min_raw = max(1e-12, 0.5 * trace_raw - sd_raw);
            float aa_cov = max(u_MinPixelCov, 0.22 / (1.0 + sqrt(max(lambda_min_raw, 1e-6))));

            float c00 = c00_raw + aa_cov;
            float c01 = c01_raw;
            float c11 = c11_raw + aa_cov;

            float det = c00 * c11 - c01 * c01;
            if (det <= 1e-12) {
                // Fallback to small isotropic splat
                c00 = 1.0;
                c01 = 0.0;
                c11 = 1.0;
                det = 1.0;
                det_raw = 1.0;
            }

            float inv_det = 1.0 / det;
            float inv00 = c11 * inv_det;
            float inv01 = -c01 * inv_det;
            float inv11 = c00 * inv_det;

            // Eigen max/min for bounding radius
            float trace = c00 + c11;
            float disc = max(0.0, 0.25 * trace * trace - det);
            float sd = sqrt(disc);
            float lambda_max = 0.5 * trace + sd;
            float lambda_min = max(1e-12, 0.5 * trace - sd);
            float radius = u_SigmaClip * sqrt(max(lambda_max, 1e-8));
            v_alpha_preserve = clamp(sqrt(max(det_raw, 1e-12) / max(det, 1e-12)), 0.05, 1.0);

            float point_size = clamp(2.0 * radius, 1.0, u_MaxPointSize);
            gl_PointSize = point_size;

            v_color = color;
            v_opacity_raw = opacity;
            float radius_draw = 0.5 * point_size;

            vec3 conic = vec3(inv00, inv01, inv11);

            // If clamped by MaxPointSize and distortion allowed: fit covariance into the
            // smaller square. Use anisotropic fit: clamp only the major axis until the
            // minor axis would exceed the limit (then clamp both).
            if (u_AllowDistortion > 0.5 && point_size + 1e-3 < 2.0 * radius) {
                float target_lmax = (radius_draw / max(u_SigmaClip, 1e-4));
                target_lmax *= target_lmax;

                float lmax2 = min(lambda_max, target_lmax);
                float lmin2 = lambda_min;
                if (lmin2 > target_lmax) {
                    // Once minor axis hits the limit, clamp both (becomes round-ish).
                    lmin2 = target_lmax;
                }

                // Eigenvector angle for symmetric 2x2
                float theta = 0.5 * atan(2.0 * c01, (c00 - c11));
                float cs = cos(theta);
                float sn = sin(theta);

                // Reconstruct clamped covariance C' = R diag(lmax2,lmin2) R^T
                float c00p = cs * cs * lmax2 + sn * sn * lmin2;
                float c11p = sn * sn * lmax2 + cs * cs * lmin2;
                float c01p = cs * sn * (lmax2 - lmin2);

                float detp = c00p * c11p - c01p * c01p;
                if (detp > 1e-20) {
                    float inv_detp = 1.0 / detp;
                    conic = vec3(c11p * inv_detp, -c01p * inv_detp, c00p * inv_detp);
                }
            }

            v_conic = conic;
            v_radius = radius_draw;

            if (u_DebugMode > 0.5) {
                if (u_DebugMode < 1.5) {
                    float ratio = sqrt(lambda_max / max(lambda_min, 1e-12));
                    float t = clamp((ratio - 1.0) / 6.0, 0.0, 1.0);
                    v_color = vec3(t, 0.15, 1.0 - t);
                } else {
                    float clamped = (point_size + 1e-3 < 2.0 * radius) ? 1.0 : 0.0;
                    v_color = mix(vec3(0.1, 0.9, 0.1), vec3(1.0, 0.1, 0.1), clamped);
                }
            }
        }
        """
    )

    create_info.fragment_source(
        """
        float sigmoid(float x)
        {
            // numerically stable sigmoid
            if (x >= 0.0) {
                float z = exp(-x);
                return 1.0 / (1.0 + z);
            }
            float z = exp(x);
            return z / (1.0 + z);
        }

        vec3 applyDisplayMatch(vec3 col)
        {
            col = clamp(col, 0.0, 1.0);
            if (u_ColorMatch > 0.5) {
                col = max(col - vec3(0.015), vec3(0.0));
                col = pow(col, vec3(1.08));
                col = col / (col + vec3(0.12));
                col *= vec3(1.12);
            }
            return clamp(col, 0.0, 1.0);
        }

        void main()
        {
            if (u_PointMode > 0.5) {
                vec2 uv = gl_PointCoord * 2.0 - 1.0;
                if (dot(uv, uv) > 1.0) {
                    discard;
                }
                fragColor = vec4(v_color, 1.0);
                return;
            }

            vec2 uv = gl_PointCoord * 2.0 - 1.0;
            vec2 d = uv * v_radius; // in pixels

            float x = d.x;
            float y = d.y;
            float power = -0.5 * (v_conic.x * x * x + 2.0 * v_conic.y * x * y + v_conic.z * y * y);
            if (power < -20.0) {
                discard;
            }

            float w = exp(power);
            float base_a = (u_OpacityIsLinear > 0.5) ? clamp(v_opacity_raw, 0.0, 1.0) : sigmoid(v_opacity_raw);
            float a = base_a * w * u_AlphaMultiplier * v_alpha_preserve;
            a = clamp(a, 0.0, 1.0);
            if (u_AlphaCurve != 1.0) {
                a = 1.0 - pow(1.0 - a, max(u_AlphaCurve, 1e-3));
            }
            if (a <= (1.0 / 255.0)) {
                discard;
            }

            if (u_UseSceneDepth > 0.5) {
                vec2 uv_scr = gl_FragCoord.xy / max(u_ViewportSize, vec2(1.0));
                float scene_z = texture(u_SceneDepthTex, uv_scr).r;
                if (gl_FragCoord.z > scene_z + u_SceneDepthBias) {
                    discard;
                }
            }

            vec3 col = v_color * max(u_ColorGain, 0.0);
            float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
            col = mix(vec3(luma), col, max(u_ColorSaturation, 0.0));
            float g = max(u_ColorGamma, 1e-3);
            if (abs(g - 1.0) > 1e-4) {
                col = pow(max(col, vec3(0.0)), vec3(1.0 / g));
            }
            fragColor = vec4(applyDisplayMatch(col), a);
        }
        """
    )

    _SHADER = gpu.shader.create_from_info(create_info)
    return _SHADER

def _ensure_dummy_depth_texture():
    global _DUMMY_DEPTH_TEX
    if _DUMMY_DEPTH_TEX is not None:
        return _DUMMY_DEPTH_TEX
    import gpu

    buf = gpu.types.Buffer("FLOAT", 1, [1.0])
    _DUMMY_DEPTH_TEX = gpu.types.GPUTexture((1, 1), format="R32F", data=buf)
    return _DUMMY_DEPTH_TEX


def _capture_scene_depth_texture(width: int, height: int):
    global _SCENE_DEPTH_TEX, _SCENE_DEPTH_SIZE
    import gpu

    w = max(1, int(width))
    h = max(1, int(height))

    try:
        fb = gpu.state.active_framebuffer_get()
        if fb is None:
            return _ensure_dummy_depth_texture()
        depth_buf = fb.read_depth(0, 0, w, h)
        depth_buf.dimensions = w * h
        _SCENE_DEPTH_TEX = gpu.types.GPUTexture((w, h), format="R32F", data=depth_buf)
        _SCENE_DEPTH_SIZE = (w, h)
        return _SCENE_DEPTH_TEX
    except Exception:
        return _ensure_dummy_depth_texture()


def _ensure_batch(data: gsp_cache.GauSplaData):
    if data.gpu_batch is not None:
        return data.gpu_batch

    shader = _ensure_shader()

    from gpu_extras.batch import batch_for_shader

    data.gpu_batch = batch_for_shader(
        shader,
        "POINTS",
        {
            "position": data.positions,
            "color": data.colors,
            "opacity": data.opacity_draw,
            "scale": data.scale_raw,
            "quat": data.rot_draw,
        },
    )
    return data.gpu_batch

def _quat_to_mat3_array(quat, *, xyzw: bool):
    import numpy as np  # type: ignore

    q = quat.astype(np.float32, copy=True)
    if xyzw:
        q = q[:, [3, 0, 1, 2]]
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1e-8)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    mats = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    mats[:, :, 0] = np.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)), axis=1)
    mats[:, :, 1] = np.stack((2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)), axis=1)
    mats[:, :, 2] = np.stack((2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)), axis=1)
    return mats


def _reference_plus_key(region_data, obj, data: gsp_cache.GauSplaData, draw_stride: int, settings) -> tuple[object, ...]:
    mw_key = tuple(round(float(v), 4) for row in obj.matrix_world for v in row)
    return (
        "reference_plus",
        _view_key(region_data),
        mw_key,
        int(draw_stride),
        int(getattr(settings, "max_points", 0)),
        round(float(getattr(settings, "sigma_clip", 2.5)), 3),
        round(float(getattr(settings, "size_multiplier", 1.0)), 4),
        round(float(getattr(settings, "scene_scale", 1.0) if getattr(settings, "use_scene_scale", False) else 1.0), 4),
        bool(getattr(settings, "scale_is_log", True)),
        str(getattr(settings, "quat_order", "WXYZ")),
        bool(getattr(settings, "use_scene_depth_cull", True)),
        round(float(data.scale_log_upper or 999.0), 4),
        int(getattr(data, "point_count", 0)),
    )


def _ensure_reference_plus_shader():
    global _SHADER_REFERENCE_PLUS
    if _SHADER_REFERENCE_PLUS is not None:
        return _SHADER_REFERENCE_PLUS

    import gpu

    create_info = gpu.types.GPUShaderCreateInfo()
    _add_shader_params_ubo(create_info)
    create_info.sampler(1, "FLOAT_2D", "u_SceneDepthTex")
    # Per-splat data textures (fetched via gl_InstanceID)
    create_info.sampler(2, "FLOAT_2D", "u_SplatPosTex")
    create_info.sampler(3, "FLOAT_2D", "u_SplatColorTex")
    create_info.sampler(4, "FLOAT_2D", "u_SplatParamTex")  # opacity, scale.xyz
    create_info.sampler(5, "FLOAT_2D", "u_SplatQuatTex")
    create_info.sampler(6, "FLOAT_2D", "u_SortOrderTex")  # converted to float due to API limits

    # Only local_uv remains as a vertex attribute
    create_info.vertex_in(0, "VEC2", "local_uv")

    iface = gpu.types.GPUStageInterfaceInfo("gau_spla_reference_plus_iface")
    iface.smooth("VEC2", "v_local_uv")
    iface.smooth("VEC3", "v_color")
    iface.smooth("FLOAT", "v_opacity_raw")
    iface.smooth("FLOAT", "v_alpha_scale")
    iface.smooth("FLOAT", "v_extend")
    create_info.vertex_out(iface)
    create_info.fragment_out(0, "VEC4", "fragColor")

    create_info.vertex_source(
        """
        float sigmoid(float x)
        {
            if (x >= 0.0) {
                float z = exp(-x);
                return 1.0 / (1.0 + z);
            }
            float z = exp(x);
            return z / (1.0 + z);
        }

        mat3 quatToMat3(vec4 qWXYZ)
        {
            float w = qWXYZ.x;
            float x = qWXYZ.y;
            float y = qWXYZ.z;
            float z = qWXYZ.w;

            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float xz = x * z;
            float yz = y * z;
            float wx = w * x;
            float wy = w * y;
            float wz = w * z;

            vec3 c0 = vec3(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy));
            vec3 c1 = vec3(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx));
            vec3 c2 = vec3(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy));
            return mat3(c0, c1, c2);
        }

        // Fetch a texel from a 2D texture using a linear index
        ivec2 idx2d(int idx, int tex_width)
        {
            return ivec2(idx % tex_width, idx / tex_width);
        }

        void main()
        {
            const float alpha_threshold = 1.0 / 255.0;
            // Single-splat probing shows that using a stronger hidden footprint
            // cutoff makes weak splats far too visible compared to LFS. Keep the
            // footprint threshold aligned with the actual alpha visibility floor.
            const float footprint_threshold = alpha_threshold;

            // Read sort order to find which splat this instance corresponds to
            int sort_tex_w = textureSize(u_SortOrderTex, 0).x;
            int splat_idx = int(texelFetch(u_SortOrderTex, idx2d(gl_InstanceID, sort_tex_w), 0).r);

            // Read per-splat data from textures
            int pos_w = textureSize(u_SplatPosTex, 0).x;
            int col_w = textureSize(u_SplatColorTex, 0).x;
            int par_w = textureSize(u_SplatParamTex, 0).x;
            int qat_w = textureSize(u_SplatQuatTex, 0).x;

            vec3 position = texelFetch(u_SplatPosTex, idx2d(splat_idx, pos_w), 0).rgb;
            vec3 color    = texelFetch(u_SplatColorTex, idx2d(splat_idx, col_w), 0).rgb;
            vec4 param    = texelFetch(u_SplatParamTex, idx2d(splat_idx, par_w), 0);
            vec4 quat     = texelFetch(u_SplatQuatTex, idx2d(splat_idx, qat_w), 0);
            float opacity_val = param.x;
            vec3 scale_val = param.yzw;

            vec4 world = u_ModelMatrix * vec4(position, 1.0);
            vec4 view = u_ViewMatrix * world;
            if (u_Isolation < 0.5 && u_IsPerspective > 0.5 && view.z >= -1e-4) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                v_local_uv = vec2(0.0);
                v_color = vec3(0.0);
                v_opacity_raw = 0.0;
                v_alpha_scale = 1.0;
                v_extend = 0.0;
                return;
            }

            mat3 M = mat3(u_ModelMatrix);
            vec3 objS = vec3(length(M[0]), length(M[1]), length(M[2]));
            mat3 Robj = mat3(normalize(M[0]), normalize(M[1]), normalize(M[2]));
            mat3 Rview = mat3(u_ViewMatrix);

            vec4 q = quat;
            if (u_Isolation > 0.5) q = vec4(1.0, 0.0, 0.0, 0.0);
            if (u_QuatOrderXYZW > 0.5) q = vec4(quat.w, quat.x, quat.y, quat.z);
            q /= max(length(q), 1e-8);
            mat3 Rq = quatToMat3(q);

            float use_log = (u_Isolation > 0.5) ? 1.0 : u_ScaleIsLog;
            vec3 scale_eval = scale_val;
            if (use_log > 0.5 && u_LogScaleUpper < 100.0) {
                scale_eval = min(scale_eval, vec3(u_LogScaleUpper));
            }
            vec3 s = (use_log > 0.5) ? exp(scale_eval) : scale_eval;
            if (u_Isolation > 0.5) s = clamp(abs(s), vec3(1e-6), vec3(1e3));
            
            s *= (u_ScaleMultiplier * u_SceneScale);
            s *= objS;

            mat3 Rcam = Rview * Robj * Rq;
            vec3 r0 = Rcam[0];
            vec3 r1 = Rcam[1];
            vec3 r2 = Rcam[2];

            float sx2 = s.x * s.x;
            float sy2 = s.y * s.y;
            float sz2 = s.z * s.z;

            float fx, fy;
            if (u_IsPerspective > 0.5 && u_StableScreenScale > 0.5) {
                float f = 0.5 * u_ViewportSize.y / max(tan(0.5 * u_StableFovRad), 1e-4);
                fx = f; fy = f;
            } else {
                fx = u_ProjectionMatrix[0][0] * u_ViewportSize.x * 0.5;
                fy = u_ProjectionMatrix[1][1] * u_ViewportSize.y * 0.5;
            }

            float z = max(1e-4, -view.z);
            float invz = 1.0 / z;
            float invz2 = invz * invz;

            float c00_raw, c01_raw, c11_raw;
            float aa_cov;
            if (u_IsPerspective > 0.5) {
                vec3 j0 = vec3(fx * invz, 0.0, fx * view.x * invz2);
                vec3 j1 = vec3(0.0, fy * invz, fy * view.y * invz2);

                float a0 = dot(j0, r0), a1 = dot(j0, r1), a2 = dot(j0, r2);
                float b0 = dot(j1, r0), b1 = dot(j1, r1), b2 = dot(j1, r2);

                c00_raw = sx2 * a0 * a0 + sy2 * a1 * a1 + sz2 * a2 * a2;
                c01_raw = sx2 * a0 * b0 + sy2 * a1 * b1 + sz2 * a2 * b2;
                c11_raw = sx2 * b0 * b0 + sy2 * b1 * b1 + sz2 * b2 * b2;

                float det_raw_tmp = max(c00_raw * c11_raw - c01_raw * c01_raw, 1e-12);
                float trace_raw_tmp = c00_raw + c11_raw;
                float sd_raw_tmp = sqrt(max(0.0, 0.25 * trace_raw_tmp * trace_raw_tmp - det_raw_tmp));
                float lambda_min_raw = max(1e-12, 0.5 * trace_raw_tmp - sd_raw_tmp);
                aa_cov = max(u_MinPixelCov, 0.22 / (1.0 + sqrt(max(lambda_min_raw, 1e-6))));
            } else {
                // Orthographic view can use a closer approximation of LFS's
                // ray-space distance metric than the Jacobian covariance path.
                float zref = max(u_ViewDistance, 1e-4);
                float k = (u_OrthoDistanceScale > 0.5) ? (1.0 / zref) : 1.0;
                vec3 ray_dir = vec3(0.0, 0.0, 1.0);
                vec3 screen_dx = vec3(1.0 / max(fx * k, 1e-6), 0.0, 0.0);
                vec3 screen_dy = vec3(0.0, 1.0 / max(fy * k, 1e-6), 0.0);

                vec3 inv_scale = 1.0 / max(abs(s), vec3(1e-8));
                vec3 A_ray = vec3(dot(r0, ray_dir), dot(r1, ray_dir), dot(r2, ray_dir)) * inv_scale;
                float A_ray_len2 = dot(A_ray, A_ray);
                if (A_ray_len2 <= 1e-12) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                    v_local_uv = vec2(0.0);
                    v_color = vec3(0.0);
                    v_opacity_raw = opacity_val;
                    v_alpha_scale = 1.0;
                    v_extend = 0.0;
                    return;
                }

                vec3 ray_hat = A_ray * inversesqrt(A_ray_len2);
                vec3 A_dx = vec3(dot(r0, screen_dx), dot(r1, screen_dx), dot(r2, screen_dx)) * inv_scale;
                vec3 A_dy = vec3(dot(r0, screen_dy), dot(r1, screen_dy), dot(r2, screen_dy)) * inv_scale;
                vec3 proj_dx = A_dx - ray_hat * dot(ray_hat, A_dx);
                vec3 proj_dy = A_dy - ray_hat * dot(ray_hat, A_dy);

                float conic00 = dot(proj_dx, proj_dx);
                float conic01 = dot(proj_dx, proj_dy);
                float conic11 = dot(proj_dy, proj_dy);
                float conic_det = conic00 * conic11 - conic01 * conic01;
                if (conic_det <= 1e-20) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                    v_local_uv = vec2(0.0);
                    v_color = vec3(0.0);
                    v_opacity_raw = opacity_val;
                    v_alpha_scale = 1.0;
                    v_extend = 0.0;
                    return;
                }

                c00_raw = conic11 / conic_det;
                c01_raw = -conic01 / conic_det;
                c11_raw = conic00 / conic_det;
                aa_cov = max(u_MinPixelCov, 0.3);
            }

            float det_raw = max(c00_raw * c11_raw - c01_raw * c01_raw, 1e-12);
            float trace_raw = c00_raw + c11_raw;
            float sd_raw = sqrt(max(0.0, 0.25 * trace_raw * trace_raw - det_raw));

            float c00 = c00_raw + aa_cov;
            float c01 = c01_raw;
            float c11 = c11_raw + aa_cov;

            float det = c00 * c11 - c01 * c01;
            if (det <= 1e-12) {
                c00 = 1.0; c01 = 0.0; c11 = 1.0; det = 1.0;
            }

            float trace = c00 + c11;
            float disc = max(0.0, 0.25 * trace * trace - det);
            float sd = sqrt(disc);
            float lambda_max = 0.5 * trace + sd;
            float lambda_min = max(1e-12, 0.5 * trace - sd);

            float sigma = max(u_SigmaClip, 1e-4);
            float compensation = clamp(sqrt(max(det_raw, 1e-12) / max(det, 1e-12)), 0.0, 1.0);
            float base_alpha = (u_OpacityIsLinear > 0.5) ? clamp(opacity_val, 0.0, 1.0) : sigmoid(opacity_val);
            float effective_alpha = clamp(base_alpha * u_AlphaMultiplier * compensation, 0.0, 1.0);
            if (effective_alpha <= alpha_threshold) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                v_local_uv = vec2(0.0);
                v_color = vec3(0.0);
                v_opacity_raw = opacity_val;
                v_alpha_scale = compensation;
                v_extend = 0.0;
                return;
            }

            float effective_extend = min(sigma, sqrt(max(0.0, 2.0 * log(effective_alpha / footprint_threshold))));
            if (effective_extend <= 1e-4) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                v_local_uv = vec2(0.0);
                v_color = vec3(0.0);
                v_opacity_raw = opacity_val;
                v_alpha_scale = compensation;
                v_extend = 0.0;
                return;
            }

            float axis_major_len = effective_extend * sqrt(max(lambda_max, 1e-8));
            float axis_minor_len = effective_extend * sqrt(max(lambda_min, 1e-8));

            vec2 axis_major = vec2(1.0, 0.0);
            vec2 axis_minor = vec2(0.0, 1.0);
            
            float diff = c00 - c11;
            if (abs(c01) > 1e-6 || abs(diff) > 1e-6) {
                float theta = 0.5 * atan(2.0 * c01, diff);
                float cs = cos(theta);
                float sn = sin(theta);
                axis_major = vec2(cs, sn);
                axis_minor = vec2(-sn, cs);
            }

            float max_axis_px = min(max(u_ViewportSize.x, u_ViewportSize.y) * 0.5,
                                    max(8.0, 0.5 * u_MaxPointSize));
            float axis_max_len = max(axis_major_len, axis_minor_len);
            float clamp_scale = min(1.0, max_axis_px / max(axis_max_len, 1e-6));
            
            vec2 offset_px = axis_major * (axis_major_len * clamp_scale * local_uv.x) + 
                             axis_minor * (axis_minor_len * clamp_scale * local_uv.y);
            
            vec2 offset_ndc = 2.0 * offset_px / max(u_ViewportSize, vec2(1.0));
            
            vec4 proj = u_ProjectionMatrix * view;
            gl_Position = proj;
            gl_Position.xy += offset_ndc * proj.w;

            v_local_uv = local_uv;
            v_opacity_raw = opacity_val;
            v_alpha_scale = compensation;
            v_extend = effective_extend;
            
            v_color = color;
            if (u_DebugMode > 0.5) {
                if (u_DebugMode < 1.5) {
                    float ratio = sqrt(lambda_max / max(lambda_min, 1e-12));
                    float t = clamp((ratio - 1.0) / 6.0, 0.0, 1.0);
                    v_color = vec3(t, 0.15, 1.0 - t);
                } else {
                    float point_size = 2.0 * effective_extend * sqrt(max(lambda_max, 1e-8));
                    float clamped = (point_size + 1e-3 < u_MaxPointSize) ? 0.0 : 1.0;
                    v_color = mix(vec3(0.1, 0.9, 0.1), vec3(1.0, 0.1, 0.1), clamped);
                }
            }
        }
        """
    )

    create_info.fragment_source(
        """
        float sigmoid(float x)
        {
            if (x >= 0.0) {
                float z = exp(-x);
                return 1.0 / (1.0 + z);
            }
            float z = exp(x);
            return z / (1.0 + z);
        }

        vec3 applyDisplayMatch(vec3 col)
        {
            col = clamp(col, 0.0, 1.0);
            if (u_ColorMatch > 0.5) {
                col = max(col - vec3(0.015), vec3(0.0));
                col = pow(col, vec3(1.08));
                col = col / (col + vec3(0.12));
                col *= vec3(1.12);
            }
            return clamp(col, 0.0, 1.0);
        }

        void main()
        {
            const float alpha_threshold = 1.0 / 255.0;

            float local2 = dot(v_local_uv, v_local_uv);
            if (local2 > 1.0) {
                discard;
            }

            float extend = max(v_extend, 1e-4);
            float power = -0.5 * extend * extend * local2;
            if (power < -20.0) {
                discard;
            }

            float w = exp(power);
            float base_a = (u_OpacityIsLinear > 0.5) ? clamp(v_opacity_raw, 0.0, 1.0) : sigmoid(v_opacity_raw);
            float a = clamp(base_a * w * u_AlphaMultiplier * v_alpha_scale, 0.0, 1.0);
            if (u_AlphaCurve != 1.0) {
                a = 1.0 - pow(1.0 - a, max(u_AlphaCurve, 1e-3));
            }
            if (a <= alpha_threshold) {
                discard;
            }

            if (u_UseSceneDepth > 0.5) {
                vec2 uv_scr = gl_FragCoord.xy / max(u_ViewportSize, vec2(1.0));
                float scene_z = texture(u_SceneDepthTex, uv_scr).r;
                if (gl_FragCoord.z > scene_z + u_SceneDepthBias) {
                    discard;
                }
            }

            vec3 col = v_color * max(u_ColorGain, 0.0);
            float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
            col = mix(vec3(luma), col, max(u_ColorSaturation, 0.0));
            float g = max(u_ColorGamma, 1e-3);
            if (abs(g - 1.0) > 1e-4) {
                col = pow(max(col, vec3(0.0)), vec3(1.0 / g));
            }
            vec3 final_col = applyDisplayMatch(col);
            fragColor = vec4(final_col, a);
        }
        """
    )

    _SHADER_REFERENCE_PLUS = gpu.shader.create_from_info(create_info)
    return _SHADER_REFERENCE_PLUS


# ---------- Texture helpers for Reference+ instancing ----------

_REFPLUS_TEXTURES = {}  # keyed by data id -> dict of GPUTexture
_REFPLUS_QUAD_BATCH = None
_REFPLUS_SORT_TEX = {}  # keyed by data id -> GPUTexture for sort order


def _tex_width_height(n: int) -> tuple:
    """Compute 2D texture dimensions to hold n texels."""
    import math
    w = min(n, 4096)
    h = math.ceil(n / w)
    return w, h


def _ensure_splat_textures(data):
    """Upload position/color/opacity+scale/quat as GPU textures. Done once per PLY load."""
    import gpu
    import numpy as np

    data_id = id(data)
    key = (data_id, int(getattr(data, "point_count", 0)))
    cached = _REFPLUS_TEXTURES.get(data_id)
    if cached is not None and cached.get("key") == key:
        return cached

    n = int(data.positions.shape[0])
    w, h = _tex_width_height(n)
    total = w * h

    def pad(arr, cols):
        """Pad array to total rows, ensuring float32."""
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if a.shape[0] < total:
            padded = np.zeros((total, cols), dtype=np.float32)
            padded[:a.shape[0], :a.shape[1]] = a[:, :cols]
            return padded
        return a[:total, :cols]

    # Position texture (RGB32F)
    pos_data = pad(data.positions, 3)
    # Add alpha channel for RGBA
    pos_rgba = np.zeros((total, 4), dtype=np.float32)
    pos_rgba[:, :3] = pos_data
    pos_buf = gpu.types.Buffer('FLOAT', total * 4, pos_rgba.flatten().tolist())
    pos_tex = gpu.types.GPUTexture((w, h), format='RGBA32F', data=pos_buf)

    # Color texture (RGB32F)
    col_data = pad(data.colors, 3)
    col_rgba = np.zeros((total, 4), dtype=np.float32)
    col_rgba[:, :3] = col_data
    col_buf = gpu.types.Buffer('FLOAT', total * 4, col_rgba.flatten().tolist())
    col_tex = gpu.types.GPUTexture((w, h), format='RGBA32F', data=col_buf)

    # Param texture: opacity (x), scale (yzw)
    opacity = pad(data.opacity_draw, 1)
    scale = pad(data.scale_raw, 3)
    param_data = np.zeros((total, 4), dtype=np.float32)
    param_data[:, 0] = opacity[:, 0]
    param_data[:, 1:4] = scale
    param_buf = gpu.types.Buffer('FLOAT', total * 4, param_data.flatten().tolist())
    param_tex = gpu.types.GPUTexture((w, h), format='RGBA32F', data=param_buf)

    # Quat texture (RGBA32F)
    quat_data = pad(data.rot_draw, 4)
    quat_buf = gpu.types.Buffer('FLOAT', total * 4, quat_data.flatten().tolist())
    quat_tex = gpu.types.GPUTexture((w, h), format='RGBA32F', data=quat_buf)

    result = {
        "key": key,
        "pos": pos_tex,
        "col": col_tex,
        "param": param_tex,
        "quat": quat_tex,
        "n": n,
        "w": w,
        "h": h,
    }
    _REFPLUS_TEXTURES[data_id] = result
    return result


def _ensure_sort_order_texture(data, order):
    """Upload the sort order as an integer texture."""
    import gpu
    import numpy as np

    n = int(order.shape[0])
    w, h = _tex_width_height(n)
    total = w * h

    order_padded = np.zeros(total, dtype=np.float32)
    order_padded[:n] = order

    order_buf = gpu.types.Buffer('FLOAT', total, order_padded.tolist())
    sort_tex = gpu.types.GPUTexture((w, h), format='R32F', data=order_buf)
    
    data_id = id(data)
    _REFPLUS_SORT_TEX[data_id] = sort_tex
    return sort_tex


def _ensure_quad_batch_refplus():
    """Create the single 6-vertex quad VBO (done once, reused forever)."""
    global _REFPLUS_QUAD_BATCH
    if _REFPLUS_QUAD_BATCH is not None:
        return _REFPLUS_QUAD_BATCH

    import gpu
    from gpu_extras.batch import batch_for_shader

    shader = _ensure_reference_plus_shader()
    corners = [
        (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0),
        (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0),
    ]
    _REFPLUS_QUAD_BATCH = batch_for_shader(shader, "TRIS", {"local_uv": corners})
    return _REFPLUS_QUAD_BATCH


def _ensure_batch_reference_plus(data, *, obj, region_data, settings, draw_stride: int, viewport_size):
    """Prepare Reference+ for rendering via draw_instanced + GPUTextures."""
    ref_key = _reference_plus_key(region_data, obj, data, draw_stride, settings)
    if data.gpu_batch_refplus is not None and data.gpu_refplus_key == ref_key:
        return data.gpu_batch_refplus

    try:
        import numpy as np
    except Exception:
        data.gpu_batch_refplus = None
        data.gpu_refplus_key = None
        return _ensure_sorted_batch(data, mv_mat=(region_data.view_matrix @ obj.matrix_world), sort_key=ref_key)

    if not hasattr(data.positions, "shape") or int(data.positions.shape[0]) == 0:
        data.gpu_batch_refplus = None
        data.gpu_refplus_key = None
        return None

    # Ensure splat data is uploaded to GPU textures (one-time cost)
    textures = _ensure_splat_textures(data)

    # Z-sort
    mw = np.asarray(obj.matrix_world, dtype=np.float64).reshape((4, 4))
    vm = np.asarray(region_data.view_matrix, dtype=np.float64).reshape((4, 4))
    mv = vm @ mw

    positions = data.positions.astype(np.float32, copy=False)
    z = (positions[:, 0] * float(mv[2, 0]) +
         positions[:, 1] * float(mv[2, 1]) +
         positions[:, 2] * float(mv[2, 2]) +
         float(mv[2, 3]))
    order = np.argsort(z).astype(np.int32, copy=False)

    n_visible = int(order.shape[0])
    if n_visible == 0:
        data.gpu_batch_refplus = None
        data.gpu_refplus_key = None
        return None

    # Upload sort order texture (~3MB for 800K points)
    sort_tex = _ensure_sort_order_texture(data, order)

    # The quad batch is shared (6 vertices, created once)
    quad_batch = _ensure_quad_batch_refplus()

    # Store all data needed for draw_instanced
    data.gpu_batch_refplus = quad_batch
    data.gpu_refplus_textures = textures
    data.gpu_refplus_sort_tex = sort_tex
    data.gpu_refplus_instance_count = n_visible
    data.gpu_refplus_key = ref_key
    data.gpu_refplus_last_time = time.perf_counter()
    data.runtime_stats.update(
        {
            "clamp_ratio": 0.0,
            "tiny_ratio": 0.0,
            "filtered_count": int(getattr(data, "filtered_count", 0)),
            "mode_path": "Quality GPU Ellipse",
            "bound_type": "GPU instanced ellipse quad",
            "compositing": "Strict sorted alpha",
            "aa_mode": "Ellipse low-pass",
            "sort_active": True,
            "rendered_count": n_visible,
        }
    )
    return data.gpu_batch_refplus


def _bind_reference_plus_instance_textures(shader, data) -> int:
    tex_data = getattr(data, "gpu_refplus_textures", None)
    sort_tex = getattr(data, "gpu_refplus_sort_tex", None)
    inst_count = int(getattr(data, "gpu_refplus_instance_count", 0))
    if not tex_data or sort_tex is None or inst_count <= 0:
        return 0

    shader.uniform_sampler("u_SplatPosTex", tex_data["pos"])
    shader.uniform_sampler("u_SplatColorTex", tex_data["col"])
    shader.uniform_sampler("u_SplatParamTex", tex_data["param"])
    shader.uniform_sampler("u_SplatQuatTex", tex_data["quat"])
    shader.uniform_sampler("u_SortOrderTex", sort_tex)
    return inst_count


def _view_key(region_data) -> tuple[float, ...]:
    # Quantized key to avoid re-sorting on tiny camera jitter
    vm = region_data.view_matrix
    pm = region_data.window_matrix

    def q(x: float) -> float:
        return round(float(x), 3)

    out = []
    for m in (vm, pm):
        for row in m:
            for v in row:
                out.append(q(v))
    return tuple(out)


def _ensure_sorted_batch(data: gsp_cache.GauSplaData, *, mv_mat, sort_key):
    if data.gpu_batch_sorted is not None and data.gpu_sort_key == sort_key:
        return data.gpu_batch_sorted

    try:
        import numpy as np  # type: ignore
    except Exception:
        # Without numpy sorting is too slow; fallback to unsorted.
        data.gpu_batch_sorted = None
        data.gpu_sort_key = None
        return _ensure_batch(data)

    pos = data.positions
    if not hasattr(pos, "shape"):
        # list fallback, too slow to sort here
        data.gpu_batch_sorted = None
        data.gpu_sort_key = None
        return _ensure_batch(data)

    # Compute view-space z for sorting (back-to-front).
    N = int(pos.shape[0])
    if N <= 1:
        data.gpu_batch_sorted = None
        data.gpu_sort_key = None
        return _ensure_batch(data)

    mv = np.asarray(mv_mat, dtype=np.float32).reshape((4, 4))
    pos_f = pos.astype(np.float32, copy=False)
    # z = dot([x,y,z,1], mv_row2)  -> far (more negative) first
    z = (
        pos_f[:, 0] * mv[2, 0]
        + pos_f[:, 1] * mv[2, 1]
        + pos_f[:, 2] * mv[2, 2]
        + mv[2, 3]
    )
    order = np.argsort(z).astype(np.int32, copy=False)

    shader = _ensure_shader()
    from gpu_extras.batch import batch_for_shader

    try:
        data.gpu_batch_sorted = batch_for_shader(
            shader,
            "POINTS",
                {
                    "position": data.positions[order],
                    "color": data.colors[order],
                    "opacity": data.opacity_draw[order],
                    "scale": data.scale_raw[order],
                    "quat": data.rot_draw[order],
                },
            )
    except Exception:
        try:
            data.gpu_batch_sorted = batch_for_shader(
                shader,
                "POINTS",
                {
                    "position": data.positions[order.tolist()],
                    "color": data.colors[order.tolist()],
                    "opacity": data.opacity_draw[order.tolist()],
                    "scale": data.scale_raw[order.tolist()],
                    "quat": data.rot_draw[order.tolist()],
                },
            )
        except Exception:
            data.gpu_batch_sorted = None
            data.gpu_sort_key = None
            return _ensure_batch(data)
    data.gpu_sort_key = sort_key
    data.gpu_sort_last_time = time.perf_counter()
    return data.gpu_batch_sorted


def _iter_visible_gau_spla_objects(context):
    for obj in context.visible_objects:
        if not hasattr(obj, OBJECT_SETTINGS_ATTR):
            continue
        s = obj.gau_spla_lfs
        if not s.is_gauspla:
            continue
        if not s.filepath:
            continue
        yield obj


def _draw():
    try:
        context = bpy.context
        if context.region_data is None:
            return

        # Robustly obtain the WINDOW region (bpy.context.area is not reliable inside handlers).
        region = context.region
        if region is None or getattr(region, "type", None) != "WINDOW":
            region = None
            try:
                area = context.area
                if area is not None and getattr(area, "type", None) == "VIEW_3D":
                    for r in area.regions:
                        if r.type == "WINDOW":
                            region = r
                            break
            except Exception:
                region = None

        if region is None:
            # Fallback: pick first VIEW_3D window region from the active window.
            try:
                win = context.window
                scr = win.screen if win else None
                if scr:
                    for area in scr.areas:
                        if area.type != "VIEW_3D":
                            continue
                        for r in area.regions:
                            if r.type == "WINDOW":
                                region = r
                                break
                        if region is not None:
                            break
            except Exception:
                region = None
        if region is None:
            _set_last_error("No VIEW_3D WINDOW region in draw callback")
            return
    except Exception as e:
        _set_last_error(f"Draw context setup failed: {e}")
        return

    _update_perf()

    import gpu

    try:
        shader = _ensure_shader()
    except Exception as e:
        _set_last_error(f"Shader creation failed: {e}")
        return

    view_mat = context.region_data.view_matrix
    proj_mat = context.region_data.window_matrix
    viewport_size = (float(region.width), float(region.height))
    view_persp = getattr(context.region_data, "view_perspective", "PERSP")
    is_perspective = 0.0 if str(view_persp) == "ORTHO" else 1.0
    view_distance = float(getattr(context.region_data, "view_distance", 1.0))

    gpu.state.blend_set("ALPHA")
    try:
        gpu.state.program_point_size_set(True)
    except Exception:
        pass
    try:
        _set_last_error(None)
        need_scene_depth_tex = False
        visible_objs = list(_iter_visible_gau_spla_objects(context))
        for obj in visible_objs:
            s = obj.gau_spla_lfs
            if (not bool(getattr(s, "isolation_mode", False))) and bool(s.enabled) and bool(getattr(s, "use_scene_depth_cull", True)):
                need_scene_depth_tex = True
                break

        scene_depth_tex = _ensure_dummy_depth_texture()
        if need_scene_depth_tex:
            scene_depth_tex = _capture_scene_depth_texture(int(region.width), int(region.height))

        for obj in visible_objs:
            s = obj.gau_spla_lfs
            ensure_look_preset_initialized(s)
            path = bpy.path.abspath(s.filepath)
            data = gsp_cache.load(path, max_points=s.max_points, force=False)
            if data.error or data.point_count == 0:
                continue

            isolation = bool(getattr(s, "isolation_mode", False))
            if (not isolation) and ((not s.enabled) and (not s.point_preview)):
                continue

            if isolation:
                gpu.state.depth_test_set("NONE")
                try:
                    gpu.state.depth_mask_set(False)
                except Exception:
                    pass
            reference_plus = bool(s.enabled)
            use_depth = (not isolation) and (not reference_plus) and bool(s.depth_test)
            if use_depth:
                depth_mode = "LESS_EQUAL"
                if getattr(s, "depth_func", "LESS_EQUAL") == "GREATER_EQUAL":
                    depth_mode = "GREATER_EQUAL"
                try:
                    gpu.state.depth_test_set(depth_mode)
                except Exception:
                    gpu.state.depth_test_set("LESS_EQUAL")
            elif not isolation:
                gpu.state.depth_test_set("NONE")
            try:
                depth_write = (not reference_plus) and bool(s.depth_write)
                gpu.state.depth_mask_set(depth_write)
            except Exception:
                pass

            mw_key = tuple(round(float(v), 4) for row in obj.matrix_world for v in row)
            color_key = ("color", _view_key(context.region_data), mw_key, int(data.sh_degree))
            gsp_cache.ensure_view_colors(
                data,
                model_matrix=obj.matrix_world,
                view_matrix=context.region_data.view_matrix,
                color_key=color_key,
            )

            batch = None
            if reference_plus:
                dyn_stride, eff_mps = _effective_stride_and_mps(obj, s, do_sort=False)
                eff_stride = 1 if isolation else max(1, int(dyn_stride))

                # Throttle: reuse cached batch during fast camera rotation
                now = time.perf_counter()
                refplus_interval = float(getattr(s, "sort_min_interval", 0.05))
                if (
                    data.gpu_batch_refplus is not None
                    and refplus_interval > 0.0
                    and (now - float(getattr(data, "gpu_refplus_last_time", 0.0))) < refplus_interval
                ):
                    batch = data.gpu_batch_refplus
                else:
                    batch = _ensure_batch_reference_plus(
                        data,
                        obj=obj,
                        region_data=context.region_data,
                        settings=s,
                        draw_stride=eff_stride,
                        viewport_size=viewport_size,
                    )
            else:
                batch = _ensure_batch(data)
                eff_stride, eff_mps = _effective_stride_and_mps(obj, s, do_sort=False)
            if batch is None:
                continue
            max_point_size = float(eff_mps) if getattr(s, "limit_point_size", True) else 16384.0
            if isolation:
                eff_stride = 1
                max_point_size = 512.0
            use_scene_depth = (not isolation) and bool(s.enabled) and bool(getattr(s, "use_scene_depth_cull", True))
            stats_key = ("stats", _view_key(context.region_data), mw_key, round(float(max_point_size), 2), int(eff_stride), bool(s.scale_is_log))
            if not reference_plus:
                gsp_cache.update_runtime_stats(
                    data,
                    model_matrix=obj.matrix_world,
                    view_matrix=context.region_data.view_matrix,
                    projection_matrix=context.region_data.window_matrix,
                    viewport_size=viewport_size,
                    scale_is_log=bool(s.scale_is_log),
                    size_multiplier=float(s.size_multiplier),
                    scene_scale=(float(s.scene_scale) if bool(s.use_scene_scale) else 1.0),
                    sigma_clip=float(s.sigma_clip),
                    max_point_size=float(max_point_size),
                    stats_key=stats_key,
                )
                data.runtime_stats.update(
                    {
                        "mode_path": "Points Preview",
                        "bound_type": "Point sprite",
                        "compositing": "Depth-tested alpha",
                        "aa_mode": "Adaptive covariance",
                        "sort_active": False,
                    }
                )
            data.runtime_stats["color_match"] = color_match_label(getattr(s, "color_match", "NEUTRAL"))
            active_shader = shader
            if reference_plus:
                active_shader = _ensure_reference_plus_shader()
            active_shader.bind()
            _bind_shader_params(
                active_shader,
                _build_shader_params(
                    view_mat=view_mat,
                    proj_mat=proj_mat,
                    model_mat=obj.matrix_world,
                    viewport_size=viewport_size,
                    is_perspective=float(is_perspective),
                    view_distance=float(view_distance),
                    settings=s,
                    draw_stride=eff_stride,
                    max_point_size=max_point_size,
                    scale_log_upper=data.scale_log_upper,
                    point_mode=(0.0 if (isolation or s.enabled) else (1.0 if s.point_preview else 0.0)),
                    isolation=isolation,
                    use_scene_depth=use_scene_depth,
                ),
            )
            active_shader.uniform_sampler("u_SceneDepthTex", scene_depth_tex if use_scene_depth else _ensure_dummy_depth_texture())
            
            if reference_plus:
                inst_count = _bind_reference_plus_instance_textures(active_shader, data)
                if inst_count > 0:
                    batch.draw_instanced(active_shader, instance_count=inst_count)
                else:
                    batch.draw(active_shader)
            else:
                batch.draw(active_shader)
    except Exception as e:
        # Ensure unexpected exceptions don't silently blank the viewport.
        _set_last_error(f"Draw failed: {e}")
    finally:
        try:
            gpu.state.depth_mask_set(True)
        except Exception:
            pass
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")


def register():
    global _DRAW_HANDLER
    global _PARAMS_UBO
    global _PERF_LAST, _PERF_DT_EMA, _PERF_FPS
    if _DRAW_HANDLER is None:
        _DRAW_HANDLER = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")
    _PARAMS_UBO = None
    _PERF_LAST = 0.0
    _PERF_DT_EMA = 0.0
    _PERF_FPS = 0.0
    _LOD_STATE.clear()
    _set_last_error(None)


def unregister():
    global _DRAW_HANDLER, _SHADER, _SHADER_REFERENCE_PLUS
    global _SCENE_DEPTH_TEX, _SCENE_DEPTH_SIZE, _DUMMY_DEPTH_TEX, _PARAMS_UBO
    global _PERF_LAST, _PERF_DT_EMA, _PERF_FPS
    if _DRAW_HANDLER is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLER, "WINDOW")
        _DRAW_HANDLER = None
    _SHADER = None
    _SHADER_REFERENCE_PLUS = None
    _SCENE_DEPTH_TEX = None
    _SCENE_DEPTH_SIZE = (0, 0)
    _DUMMY_DEPTH_TEX = None
    _PARAMS_UBO = None
    _PERF_LAST = 0.0
    _PERF_DT_EMA = 0.0
    _PERF_FPS = 0.0
    _LOD_STATE.clear()
    _set_last_error(None)
