"""Disk V2 reference 层的对外导出接口。

本模块只负责汇总 `disk_v2` 当前已经稳定的几何函数、基础物理场函数、结构调制函数
和参数对象，作为 reference 实现的包级入口。

这里不放具体实现逻辑；真正的实现分别位于 `geometry.py`、`physical_fields.py`、
`structure_modulations.py` 和 `params.py`。
"""

from .physical_fields import (
    angular_velocity_field,
    density_field,
    midplane_density_field,
    midplane_temperature_field,
    temperature_field,
)
from .geometry import disk_half_thickness, disk_radial_mask, disk_radial_weight, disk_vertical_weight, disk_volume_mask
from .structure_modulations import (
    hotspot_modulation,
    shear_modulation,
    structure_modulation,
    weak_mode_modulation,
)
from .params import DiskV2Params, DiskV2StructureParams

__all__ = [
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
    "angular_velocity_field",
    "temperature_field",
    "weak_mode_modulation",
    "shear_modulation",
    "hotspot_modulation",
    "structure_modulation",
]
