"""Disk V2：体积吸积盘（与参考实现 `scripts/proto_disk_reference.py` 预设 M 对齐）。

模块分层：

- `params.py`：参数对象（`DiskV2Params` 盘几何、`DiskV2VolumeParams` 体积模型）。
- `physical_fields.py`：Page–Thorne 温度 / T_peak 推导、SS 外区 H(r)、Σ(r)（NumPy 参考）。
- `relativity.py`：频移 g 的 NumPy 参考与严格 GR 对照。
- `palette.py`：CIE 黑体色度 / 亮度查找表、von Kries 白平衡。
- `noise_ti.py` / `advection.py`：程序化噪声与刚体环平流。
- `taichi_impl.py`：体积密度场 `DiskV2Taichi.density_I` 与 Taichi 端频移。
- `taichi_render.py`：主光追 `DiskV2Renderer`。
- `postfx.py`：后处理链（白平衡 → bloom → 色散 → 保色度 ACES → sRGB）。

包级只导出无 Taichi 依赖的参数与参考函数；渲染器请从 `src.v2.taichi_render` 导入。
"""

from .palette import blackbody_color, blackbody_luminance, white_balance_gain
from .params import SCHWARZSCHILD_ISCO_R_S, DiskV2Params, DiskV2VolumeParams
from .physical_fields import derive_t_peak, page_thorne_flux, ss_half_thickness, ss_surface_density
from .relativity import disk_g_factor, exact_equatorial_g_factor, local_photon_direction, orbital_beta_local

__all__ = [
    "SCHWARZSCHILD_ISCO_R_S",
    "DiskV2Params",
    "DiskV2VolumeParams",
    "blackbody_color",
    "blackbody_luminance",
    "white_balance_gain",
    "derive_t_peak",
    "page_thorne_flux",
    "ss_half_thickness",
    "ss_surface_density",
    "disk_g_factor",
    "exact_equatorial_g_factor",
    "local_photon_direction",
    "orbital_beta_local",
]
