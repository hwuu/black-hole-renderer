"""Disk V2 reference 层的对外导出接口。

本模块只负责汇总 `disk_v2` 当前已经稳定的几何函数、基础物理场函数、结构调制函数
和参数对象，作为 reference 实现的包级入口。

这里不放具体实现逻辑；真正的实现分别位于 `geometry.py`、`physical_fields.py`、
`structure_modulations.py`、`palette.py` 和 `params.py`。
"""

from .physical_fields import (
    angular_velocity_field,
    density_field,
    midplane_density_field,
    midplane_temperature_field,
    raw_midplane_density_field,
    raw_midplane_temperature_field,
    temperature_field,
)
from .geometry import disk_half_thickness, disk_radial_mask, disk_radial_weight, disk_vertical_weight, disk_volume_mask
from .structure_modulations import (
    clump_modulation,
    hotspot_modulation,
    shear_modulation,
    structure_modulation,
    structure_modulation_density,
    structure_modulation_emission,
    weak_mode_modulation,
)
from .params import (
    SCHWARZSCHILD_ISCO_R_S,
    DiskV2PaletteParams,
    DiskV2Params,
    DiskV2StructureParams,
)
from .palette import (
    apply_exposure,
    apply_palette,
    blackbody_color,
    cinematic_color,
    gamma_correct,
    palette_color,
    render_hdr_to_ldr,
    tonemap,
)
from .imaging import (
    observed_visible_temperature,
    physical_baseline_flux,
    physical_baseline_volume_flux,
    reference_exposure,
    tau_effective_midplane,
)
from .relativity import (
    doppler_g_factor,
    gravitational_g_factor,
    omega_kep,
    omega_norm,
    orbital_beta_local,
    total_g_factor,
)

__all__ = [
    "SCHWARZSCHILD_ISCO_R_S",
    "DiskV2PaletteParams",
    "DiskV2Params",
    "DiskV2StructureParams",
    "disk_half_thickness",
    "disk_radial_mask",
    "disk_radial_weight",
    "disk_vertical_weight",
    "disk_volume_mask",
    "density_field",
    "midplane_density_field",
    "midplane_temperature_field",
    "raw_midplane_density_field",
    "raw_midplane_temperature_field",
    "angular_velocity_field",
    "temperature_field",
    "weak_mode_modulation",
    "shear_modulation",
    "clump_modulation",
    "hotspot_modulation",
    "structure_modulation",
    "structure_modulation_density",
    "structure_modulation_emission",
    "blackbody_color",
    "cinematic_color",
    "palette_color",
    "apply_exposure",
    "tonemap",
    "gamma_correct",
    "render_hdr_to_ldr",
    "apply_palette",
    "tau_effective_midplane",
    "physical_baseline_flux",
    "physical_baseline_volume_flux",
    "reference_exposure",
    "observed_visible_temperature",
    "omega_kep",
    "omega_norm",
    "gravitational_g_factor",
    "orbital_beta_local",
    "doppler_g_factor",
    "total_g_factor",
]
