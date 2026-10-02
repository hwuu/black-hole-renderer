#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwarzschild 黑洞光线追踪渲染器（入口）。

实现按层拆分：
- `src/core`：常量、相机、天空盒、图像输出（V1/V2 共用）
- `src/v1`：程序纹理、实体生命周期、渲染器、视频管线
- `src/v2`：体积吸积盘
- `src/cli`：命令行参数与分发

本文件保留为兼容门面与命令行入口：`python render.py ...` 用法不变，
`from render import X` 的历史引用继续可用。
"""

from src.cli import main, parse_args, validate_args
from src.core.camera import (
    build_camera,
)
from src.core.constants import (
    DISK_ALPHA_GAIN,
    DISK_COLOR_TEMPERATURE,
    DISK_GENERATION_SCALE_CHOICES,
    DISK_RADIAL_BRIGHTNESS_MAX,
    DISK_RADIAL_BRIGHTNESS_MIN,
    DISK_RADIAL_BRIGHTNESS_POWER,
    ENABLE_DISK_SPIRAL_ARMS,
    EPS,
    FILAMENT_BIRTH_FADE_DUR,
    FILAMENT_DEATH_THRESHOLD,
    FILAMENT_MAX_LIFETIME,
    FILAMENT_SHEAR_ALPHA,
    FILAMENT_TAU_COOL,
    G_BRIGHTNESS_GAIN,
    G_FACTOR_CAP,
    G_LUMINOSITY_POWER,
    RS,
    R_DISK_INNER_DEFAULT,
    R_DISK_OUTER_DEFAULT,
    SKY_GALACTIC_CENTER_GLOW,
    SKY_MILKY_WAY_GLOW,
    SKY_STAR_BRIGHTNESS_GAIN,
    SKY_STAR_BRIGHTNESS_MAX,
    SKY_STAR_BRIGHTNESS_MIN,
    SKY_STAR_COLOR_SATURATION,
    SKY_STAR_SIZE_MAX,
    SKY_STAR_SIZE_MIN,
)
from src.core.imaging import (
    save_image,
)
from src.core.skybox import (
    _blackbody_rgb,
    generate_skybox,
    load_or_generate_skybox,
    sample_skybox_bilinear,
)
from src.v1.lifecycle import (
    EntityFactory,
    EntityInstance,
    _advance_lifecycle_frame,
    _init_lifecycle_system,
)
from src.v1.pipeline import (
    render_image,
    render_interactive,
    render_video,
)
from src.v1.renderer import (
    TaichiRenderer,
)
from src.v1.texture import (
    DiskTextureRotatingState,
    _apply_disturbance,
    _blend_azimuthal_seam,
    _compose_disk_texture_from_fields,
    _compute_rotation_pixels,
    _compute_upscaled_rotation_pixels,
    _fbm_noise,
    _generate_azimuthal_hotspot,
    _generate_disk_texture_rotating_from_state,
    _generate_disturbance_mod,
    _generate_filaments,
    _generate_hotspots,
    _generate_rt_spikes,
    _generate_spiral_arms,
    _generate_temperature_base,
    _generate_turbulence,
    _periodic_pixel_noise,
    _roll_rows,
    _spawn_single_filament,
    _spawn_single_hotspot,
    _spawn_single_rt_spike,
    _tileable_noise,
    _validate_disk_generation_scale,
    build_disk_texture_rotating_state,
    compute_disk_texture_resolution,
    compute_edge_alpha,
    generate_disk_mipmaps,
    generate_disk_texture,
    generate_disk_texture_rotating,
    load_cached_disk_texture,
    load_disk_texture,
)

if __name__ == "__main__":
    main()
