from __future__ import annotations

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import PropertyGroup


_LOOK_PRESET_VALUES = {
    "REFERENCE": {
        "alpha_multiplier": 1.0,
        "alpha_curve": 1.0,
        "color_gain": 1.0,
        "color_saturation": 1.0,
        "color_gamma": 1.0,
    },
    "BALANCED": {
        "alpha_multiplier": 1.12,
        "alpha_curve": 1.18,
        "color_gain": 1.04,
        "color_saturation": 1.04,
        "color_gamma": 0.98,
    },
    "FULL": {
        "alpha_multiplier": 1.28,
        "alpha_curve": 1.34,
        "color_gain": 1.08,
        "color_saturation": 1.08,
        "color_gamma": 0.95,
    },
}

_COLOR_MATCH_LABELS = {
    "NEUTRAL": "Neutral",
    "LICHTFELD": "Soft Contrast",
}

OBJECT_SETTINGS_ATTR = "gau_spla_lfs"

_BRIDGE_STATUS_ITEMS = (
    ("NOT_LINKED", "Not Linked", "Объект не связан с watched export file"),
    ("LINKED", "Linked", "Watched export file назначен, но sync ещё не выполнялся"),
    ("WAITING_STABLE", "Waiting For Stable Write", "Файл изменился, но ещё не считается стабильно записанным"),
    ("RELOADING", "Reloading", "Идёт reload linked file"),
    ("SYNCED", "Synced", "Последний reload linked file выполнен успешно"),
    ("MISSING_FILE", "Missing File", "Watched file не найден"),
    ("RELOAD_FAILED", "Reload Failed", "Последний reload linked file завершился ошибкой"),
)
_BRIDGE_STATUS_LABELS = {key: label for key, label, _desc in _BRIDGE_STATUS_ITEMS}


def apply_look_preset(settings: "GauSplaObjectSettings", preset: str) -> None:
    values = _LOOK_PRESET_VALUES.get(str(preset), _LOOK_PRESET_VALUES["BALANCED"])
    for name, value in values.items():
        setattr(settings, name, value)
    settings.look_preset_initialized = True


def ensure_look_preset_initialized(settings: "GauSplaObjectSettings") -> None:
    if getattr(settings, "look_preset_initialized", False):
        return
    try:
        apply_look_preset(settings, "REFERENCE")
    except Exception:
        # UI draw and viewport callbacks may run in restricted contexts where
        # writing ID properties is disallowed. In that case keep current values
        # and let safe operator paths initialize the preset later.
        return


def color_match_label(value: str) -> str:
    return _COLOR_MATCH_LABELS.get(str(value), str(value))

def bridge_status_label(value: str) -> str:
    return _BRIDGE_STATUS_LABELS.get(str(value), str(value))


class GauSplaObjectSettings(PropertyGroup):
    is_gauspla: BoolProperty(
        name="Is GauSpla Object",
        default=False,
        options={"HIDDEN"},
    )

    enabled: BoolProperty(
        name="Enabled",
        default=True,
        description="Рисовать как splat. Если выключено — рисовать как обычные точки (preview)",
    )

    point_preview: BoolProperty(
        name="Point Preview",
        description="Показывать обычные точки, когда Enabled выключен",
        default=True,
    )

    ui_show_sync: BoolProperty(name="UI Sync", default=True, options={"HIDDEN"})
    ui_show_display: BoolProperty(name="UI Display", default=True, options={"HIDDEN"})
    ui_show_look: BoolProperty(name="UI Look", default=True, options={"HIDDEN"})
    look_preset_initialized: BoolProperty(name="Look Preset Initialized", default=False, options={"HIDDEN"})
    axis_fix_initialized: BoolProperty(name="Axis Fix Initialized", default=False, options={"HIDDEN"})

    filepath: StringProperty(
        name="PLY File",
        subtype="FILE_PATH",
        description="Путь к .ply с Gaussian Splat",
        default="",
    )

    bridge_watch_path: StringProperty(
        name="Watched Export",
        subtype="FILE_PATH",
        description="Экспортный .ply, который GauSpla должен отслеживать и reload-ить в этот же объект",
        default="",
    )

    bridge_auto_sync: BoolProperty(
        name="Auto Sync",
        description="Следить за watched export file и автоматически reload-ить тот же объект при обновлении файла",
        default=False,
    )

    bridge_poll_interval: FloatProperty(
        name="Poll Interval",
        description="Интервал проверки watched export file в секундах. Меньше = быстрее реакция, но выше нагрузка",
        default=0.25,
        min=0.1,
        soft_min=0.15,
        soft_max=2.0,
        precision=2,
        step=5,
    )

    bridge_status: EnumProperty(
        name="Sync Status",
        description="Последний статус file-based sync для этого объекта",
        items=_BRIDGE_STATUS_ITEMS,
        default="NOT_LINKED",
    )

    bridge_last_sig: StringProperty(name="Last Sync Signature", default="", options={"HIDDEN"})
    bridge_last_reload_ts: FloatProperty(name="Last Reload Timestamp", default=0.0, options={"HIDDEN"})
    bridge_last_error: StringProperty(name="Last Sync Error", default="", options={"HIDDEN"})

    max_points: IntProperty(
        name="Preview Load Limit",
        description="Ограничить количество загружаемых точек для превью (0 = без лимита). Меняет сам загруженный набор, а не только качество отображения",
        default=500_000,
        min=0,
        soft_max=5_000_000,
    )

    disabled_point_size: FloatProperty(
        name="Point Size",
        description="Размер точек (в пикселях), когда Enabled выключен",
        default=2.0,
        min=1.0,
        soft_max=20.0,
    )

    use_view_stabilization: BoolProperty(
        name="View Stabilization",
        description="Стабилизировать размер при переключении Persp/Ortho и изменении фокусного",
        default=False,
    )

    stable_screen_scale: BoolProperty(
        name="Stable Screen Scale",
        description="Стабилизировать размер между Ortho/Persp и изменением Focal Length (использовать фиксированный экранный FOV для масштаба)",
        default=False,
    )

    stable_fov_deg: FloatProperty(
        name="Stable FOV",
        description="Вертикальный FOV (градусы) для Stable Screen Scale",
        default=60.0,
        min=5.0,
        max=170.0,
    )

    ortho_distance_scale: BoolProperty(
        name="Ortho Distance Scale",
        description="В ортографическом виде масштабировать размер с расстоянием до камеры (чтобы меньше 'скакало' при переключении Persp/Ortho)",
        default=True,
    )

    render_mode: EnumProperty(
        name="Render Mode",
        description="Текущий релизный splat-рендер path",
        items=(
            ("REFERENCE_PLUS", "Quality", "Основной качественный режим: GPU ellipse bounds + strict sorted compositing"),
        ),
        default="REFERENCE_PLUS",
    )

    color_match: EnumProperty(
        name="Tone",
        description="Финальный display look после линейного цветового пайплайна",
        items=(
            ("NEUTRAL", "Neutral", "Нейтральный вывод без дополнительного display shaping"),
            ("LICHTFELD", "Soft Contrast", "Более глубокие тени и мягкая компрессия хайлайтов без внешних брендовых ассоциаций"),
        ),
        default="NEUTRAL",
    )

    use_scene_scale: BoolProperty(
        name="Use Scene Scale",
        description="Дополнительный множитель размера сплатов (позиции не трогает). Если выключено — размер контролируй масштабом пустышки",
        default=False,
    )

    scene_scale: FloatProperty(
        name="Splat Size Scale",
        description="Масштабирует только размер сплатов (scale_*), не влияет на позицию. Полезно для тонкой подстройки",
        default=1.0,
        min=0.000001,
        soft_min=0.001,
        soft_max=100.0,
    )

    auto_guess_scale: BoolProperty(
        name="Auto Guess Scale",
        description="При импорте/перезагрузке пытаться угадать, log ли scale_*",
        default=True,
    )

    auto_lod: BoolProperty(
        name="Auto LOD",
        description="Автоматически подбирать draw stride/лимиты под целевой FPS",
        default=True,
    )

    target_fps: IntProperty(
        name="Target FPS",
        description="Целевой FPS для авто-LOD",
        default=30,
        min=1,
        soft_max=120,
    )

    lod_hysteresis: FloatProperty(
        name="LOD Hysteresis",
        description="Гистерезис по FPS чтобы не дергалось (в FPS)",
        default=3.0,
        min=0.0,
        soft_max=20.0,
    )

    lod_update_interval: FloatProperty(
        name="LOD Update",
        description="Как часто авто-LOD может менять параметры (сек)",
        default=0.3,
        min=0.05,
        soft_max=2.0,
    )

    lod_min_stride: IntProperty(
        name="LOD Min Stride",
        description="Минимальный stride для авто-LOD",
        default=1,
        min=1,
        soft_max=50,
    )

    lod_max_stride: IntProperty(
        name="LOD Max Stride",
        description="Максимальный stride для авто-LOD",
        default=20,
        min=1,
        soft_max=200,
    )

    lod_adjust_point_size: BoolProperty(
        name="LOD Point Size",
        description="Авто-LOD может уменьшать Max Point Size при низком FPS",
        default=False,
    )

    lod_min_point_size: IntProperty(
        name="LOD Min Point Size",
        description="Нижняя граница для авто-LOD Max Point Size",
        default=128,
        min=1,
        soft_max=2048,
    )

    sort_min_interval: FloatProperty(
        name="Sort Throttle",
        description="Минимальный интервал между пересортировками (сек)",
        default=0.25,
        min=0.0,
        soft_max=2.0,
    )

    draw_stride: IntProperty(
        name="Draw Stride",
        description="Рисовать каждый N-й splat (ускоряет вьюпорт). 1 = рисовать все",
        default=1,
        min=1,
        soft_max=50,
    )

    size_multiplier: FloatProperty(
        name="Splat Scale",
        description="Доп. множитель для scale_* (после Scene Scale). Удобно для тонкой подстройки",
        default=1.0,
        min=0.0,
        soft_max=200.0,
    )

    scale_is_log: BoolProperty(
        name="Scale Is Log",
        description="Если включено, scale_* интерпретируется как log(scale) и в шейдере применяется exp()",
        default=True,
    )

    quat_order: EnumProperty(
        name="Quaternion Order",
        description="Порядок компонент rot_0..3 в PLY. Часто это WXYZ, но иногда XYZW",
        items=(
            ("WXYZ", "WXYZ", "rot=(w,x,y,z)"),
            ("XYZW", "XYZW", "rot=(x,y,z,w)"),
        ),
        default="WXYZ",
    )

    opacity_encoding: EnumProperty(
        name="Opacity Encoding",
        description="Как интерпретировать поле opacity/alpha (после импорта). AUTO = если все значения в [0..1], считаем что это alpha",
        items=(
            ("AUTO", "Auto", "Авто-определение (0..1 => Linear, иначе Logit)"),
            ("LOGIT", "Logit", "opacity_raw это logit(alpha), в шейдере применяется sigmoid()"),
            ("LINEAR", "Linear 0..1", "opacity_raw это alpha в 0..1"),
        ),
        default="AUTO",
    )

    sigma_clip: FloatProperty(
        name="Sigma Clip",
        description="Радиус спрайта в сигмах (примерно 3 = хороший компромисс)",
        default=2.2,
        min=0.5,
        soft_max=6.0,
    )

    min_pixel_cov: FloatProperty(
        name="Min Pixel Cov",
        description="Минимальная добавка к 2D ковариации (стабилизация и антиалиасинг)",
        default=0.1,
        min=0.0,
        soft_max=2.0,
    )

    max_point_size: IntProperty(
        name="Max Point Size",
        description="Ограничение gl_PointSize (в пикселях) чтобы избежать огромных спрайтов",
        default=256,
        min=1,
        soft_max=4096,
    )

    limit_point_size: BoolProperty(
        name="Limit Point Size",
        description="Ограничивать размер квадрата-контейнера (рекомендуется для скорости и чтобы не убивать fillrate)",
        default=True,
    )

    allow_splat_distortion: BoolProperty(
        name="Allow Splat Distortion",
        description="Если включено, при упоре в Max Point Size сплат будет 'ужиматься' (меняется ковариация), а не жёстко обрезаться квадратом",
        default=True,
    )

    alpha_multiplier: FloatProperty(
        name="Density",
        description="Общая плотность/непрозрачность сплатов",
        default=1.0,
        min=0.0,
        soft_max=5.0,
    )

    alpha_curve: FloatProperty(
        name="Alpha Curve",
        description="Нелинейная кривая альфы: >1 делает сплаты более 'плотными' (удобно, если всё бледное)",
        default=1.0,
        min=0.1,
        soft_max=6.0,
    )

    color_gain: FloatProperty(
        name="Exposure",
        description="Общая яркость цвета после импорта и перед display shaping",
        default=1.0,
        min=0.0,
        soft_max=4.0,
    )

    color_saturation: FloatProperty(
        name="Saturation",
        description="Насыщенность: 0=серый, 1=оригинал, >1=сочнее",
        default=1.0,
        min=0.0,
        soft_max=3.0,
    )

    color_gamma: FloatProperty(
        name="Gamma",
        description="Гамма-коррекция цвета (1=без изменений). Удобно для 'блеклых' наборов",
        default=1.0,
        min=0.2,
        soft_max=3.0,
    )

    depth_test: BoolProperty(
        name="Depth Test",
        description="Учитывать глубину (окклюзия)",
        default=True,
    )

    depth_func: EnumProperty(
        name="Depth Function",
        description="Функция теста глубины. Если видно только 'на фоне геометрии' — ставь LESS_EQUAL. Если дальние перекрывают ближние — попробуй GREATER_EQUAL",
        items=(
            ("LESS_EQUAL", "Less/Equal", "Стандартный Z (по умолчанию)"),
            ("GREATER_EQUAL", "Greater/Equal", "Reversed-Z (некоторые конфиги/рендер-пути)"),
        ),
        default="LESS_EQUAL",
    )

    depth_write: BoolProperty(
        name="Depth Write",
        description="Записывать глубину (обычно лучше выключить для прозрачных сплатов)",
        default=False,
    )

    use_scene_depth_cull: BoolProperty(
        name="Scene Depth Cull",
        description="Отсекать фрагменты сплатов за геометрией сцены по depth буферу Blender",
        default=True,
    )

    scene_depth_bias: FloatProperty(
        name="Depth Bias",
        description="Смещение depth-сравнения (больше = меньше мерцания, но чуть больше 'протечек')",
        default=0.0002,
        min=0.0,
        soft_max=0.01,
    )

    debug_mode: EnumProperty(
        name="Debug Mode",
        description="Визуальная диагностика (может замедлять)",
        items=(
            ("NONE", "Off", "Обычный рендер"),
            ("ANISO", "Anisotropy", "Цвет по анизотропии (lambda_max/lambda_min)"),
            ("CLAMP", "Clamped", "Подсветить сплаты, упёршиеся в Max Point Size"),
        ),
        default="NONE",
    )

    isolation_mode: BoolProperty(
        name="Isolation Mode (Force Visible)",
        description=(
            "Брутфорс-режим для диагностики: форсирует splat-рендер даже если выключен, "
            "отключает depth test/write и cull, игнорирует quaternion (Rq=Identity) и "
            "использует безопасную интерпретацию opacity/scale"
        ),
        default=False,
    )


_CLASSES = (GauSplaObjectSettings,)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.gau_spla_lfs = PointerProperty(type=GauSplaObjectSettings)


def unregister():
    del bpy.types.Object.gau_spla_lfs
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
