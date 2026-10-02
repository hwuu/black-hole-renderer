from dataclasses import dataclass
from typing import Tuple, List, Optional
import math
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from src.core.constants import DISK_COLOR_TEMPERATURE, DISK_GENERATION_SCALE_CHOICES, ENABLE_DISK_SPIRAL_ARMS
from src.core.skybox import _blackbody_rgb


def _validate_disk_generation_scale(generation_scale: int) -> int:
    if generation_scale not in DISK_GENERATION_SCALE_CHOICES:
        raise ValueError(
            f"disk_generation_scale must be one of {DISK_GENERATION_SCALE_CHOICES}, got {generation_scale}"
        )
    return generation_scale


def compute_edge_alpha(height: int, inner_soft: float = 0.1, outer_soft: float = 0.3) -> np.ndarray:
    """计算边缘软化的 alpha 通道"""
    v = np.linspace(0, 1, height).astype(np.float32)
    alpha = np.ones_like(v)
    inner_mask = v < inner_soft
    outer_mask = v > (1 - outer_soft)
    alpha[inner_mask] = (v[inner_mask] / inner_soft) ** 3.0
    alpha[outer_mask] = ((1 - v[outer_mask]) / outer_soft) ** 2
    return alpha


def load_disk_texture(path: Optional[str]) -> Optional[np.ndarray]:
    """加载吸积盘纹理，返回 (h, w, 4) float32 数组（RGBA，边缘软化 alpha）"""
    if path and os.path.isfile(path):
        print(f"Loading disk texture: {path}")
        img = Image.open(path).convert("RGB")
        rgb = np.array(img, dtype=np.float32) / 255.0
        h, w = rgb.shape[:2]
        alpha = compute_edge_alpha(h)[:, None].astype(np.float32)
        alpha = np.broadcast_to(alpha, (h, w)).copy()
        alpha = alpha[:, :, None]
        return np.concatenate([rgb, alpha], axis=2)
    return None


@dataclass(frozen=True)
class DiskTextureRotatingState:
    n_phi: int
    n_r: int
    seed: int
    generation_scale: int
    r_inner: float
    r_outer: float
    enable_rt: bool
    color_temp: float
    omega_rows: np.ndarray
    edge: np.ndarray
    temp_base: np.ndarray
    spiral: np.ndarray
    spiral_temp: np.ndarray
    turbulence: np.ndarray
    turb_temp: np.ndarray
    arcs: np.ndarray
    arcs_temp: np.ndarray
    rt_spikes: np.ndarray
    rt_temp: np.ndarray
    hotspot: np.ndarray
    hotspot_temp: np.ndarray
    az_hotspot: np.ndarray
    disturb_mod: np.ndarray


def _generate_temperature_base(rng: np.random.Generator, n_r: int, n_phi: int,
                               r_norm_grid: np.ndarray) -> np.ndarray:
    """生成吸积盘基础温度场（不含动态旋转）。"""
    radial_decay = np.clip(1.0 - r_norm_grid, 0, 1) ** 1.3
    temp_coarse = _fbm_noise((n_r, n_phi), rng, octaves=4, persistence=0.6, base_scale=8, wrap_u=True)
    temp_fine = _fbm_noise((n_r, n_phi), rng, octaves=5, persistence=0.45, base_scale=3, wrap_u=True)
    temp_noise = 0.6 * temp_coarse + 0.4 * temp_fine
    temp_base = np.clip(radial_decay * (0.85 + 0.15 * temp_noise), 0, 1)
    temp_base *= 0.25
    return temp_base.astype(np.float32)


def _generate_disturbance_mod(rng: np.random.Generator, n_r: int, n_phi: int,
                              kep_shift_pixels: np.ndarray, r_norm_grid: np.ndarray,
                              t_offset: float = 0.0, omega_grid: np.ndarray = None,
                              generation_scale: int = 2) -> np.ndarray:
    """生成湍流扰动调制场。"""
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]
    kep_shift_pixels_low = (kep_shift_pixels // scale_factor).astype(np.int32)[:low_n_r, :]

    disturb_coarse = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=8, freq_v=4)
    disturb_mid = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=32, freq_v=16)
    disturb_fine = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=100, freq_v=50)
    disturb_extra = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=250, freq_v=125)

    for layer in [disturb_coarse, disturb_mid, disturb_fine, disturb_extra]:
        for ri in range(low_n_r):
            layer[ri, :] = np.roll(layer[ri, :], kep_shift_pixels_low[ri, 0])

    rotation_pixels_low = None
    if t_offset != 0.0 and omega_grid is not None:
        omega_grid_low = omega_grid[::scale_factor, ::scale_factor]
        rotation_pixels_low = (t_offset * omega_grid_low / (2 * np.pi) * low_n_phi).astype(int)
        for layer in [disturb_coarse, disturb_mid, disturb_fine, disturb_extra]:
            for ri in range(low_n_r):
                layer[ri, :] = np.roll(layer[ri, :], -rotation_pixels_low[ri, 0])

    disturb_pixel = _periodic_pixel_noise((low_n_r, low_n_phi), rng)
    if rotation_pixels_low is not None:
        for ri in range(low_n_r):
            disturb_pixel[ri, :] = np.roll(disturb_pixel[ri, :], -rotation_pixels_low[ri, 0])

    disturb_mod_low = (0.05 * disturb_coarse + 0.15 * disturb_mid + 0.30 * disturb_fine
                       + 0.30 * disturb_extra + 0.20 * disturb_pixel)
    disturb_mod_low = np.clip(disturb_mod_low * 1.4, 0.05, 1.0)

    radial_preserve = 0.6 + 0.4 * low_r_norm_grid
    disturb_mod_low = np.clip(disturb_mod_low * radial_preserve, 0.1, 1.0)

    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    return np.kron(disturb_mod_low, upscale_kernel)[:n_r, :n_phi].astype(np.float32)


def _compose_disk_texture_from_fields(temp_base: np.ndarray, temp_struct: np.ndarray,
                                      density: np.ndarray, az_hotspot: np.ndarray,
                                      edge: np.ndarray, color_temp: float) -> np.ndarray:
    """从温度/密度场合成最终 RGBA 纹理。"""
    density = density * edge[:, None]
    density = np.clip(density / (np.percentile(density, 98) + 1e-6), 0, 1)

    if np.any(temp_struct > 0):
        struct_scale = np.percentile(temp_struct[temp_struct > 0], 95)
        temp_struct_scaled = temp_struct / (struct_scale + 1e-6)
    else:
        temp_struct_scaled = temp_struct
    temp_struct_scaled = np.clip(temp_struct_scaled * 0.8, 0, 1.2)

    struct_max_per_r = np.max(temp_struct_scaled, axis=1)
    struct_p70_per_r = np.quantile(temp_struct_scaled, 0.7, axis=1)
    struct_ceiling = np.maximum(struct_p70_per_r, 0.05)
    temp_base = np.minimum(temp_base, struct_ceiling[:, None])
    temp_base = np.minimum(temp_base, struct_max_per_r[:, None])

    temperature_field = np.clip(np.maximum(temp_base, temp_struct_scaled), 0, 1)

    t_factor = (color_temp - 4500) / (6500 - 2700)
    T_min = 2000 + t_factor * 1000
    T_max = 9000 + t_factor * 3000

    temp_aniso = np.clip(temperature_field * (0.9 + 0.25 * az_hotspot), 0, 1)
    T_K = T_min + temp_aniso * (T_max - T_min)
    bb_color = _blackbody_rgb(T_K)
    bb_color[:, :, 2] = np.minimum(bb_color[:, :, 2], bb_color[:, :, 0])

    luminosity = np.clip(np.sqrt(temp_aniso), 0, 1)

    tex = np.zeros((temp_base.shape[0], temp_base.shape[1], 4), dtype=np.float32)
    tex[:, :, 0] = np.clip(bb_color[:, :, 0] * luminosity, 0, 1)
    tex[:, :, 1] = np.clip(bb_color[:, :, 1] * luminosity, 0, 1)
    tex[:, :, 2] = np.clip(bb_color[:, :, 2] * luminosity, 0, 1)
    tex[:, :, 3] = np.clip(density, 0, 1)
    return tex


def _roll_rows(field: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """按行循环平移二维/三维场。"""
    rolled = np.empty_like(field)
    if field.ndim == 2:
        for ri, shift in enumerate(shifts):
            rolled[ri, :] = np.roll(field[ri, :], -int(shift))
        return rolled
    if field.ndim == 3:
        for ri, shift in enumerate(shifts):
            rolled[ri, :, :] = np.roll(field[ri, :, :], -int(shift), axis=0)
        return rolled
    raise ValueError(f"Unsupported field ndim: {field.ndim}")


def _compute_rotation_pixels(omega_rows: np.ndarray, t_offset: float, n_phi: int) -> np.ndarray:
    return (t_offset * omega_rows / (2 * np.pi) * n_phi).astype(np.int32)


def _compute_upscaled_rotation_pixels(omega_rows: np.ndarray, t_offset: float, n_phi: int,
                                      scale_factor: int = 2) -> np.ndarray:
    scale_factor = _validate_disk_generation_scale(scale_factor)
    low_n_phi = n_phi // scale_factor
    low_omega_rows = omega_rows[::scale_factor]
    low_shifts = (t_offset * low_omega_rows / (2 * np.pi) * low_n_phi).astype(np.int32)
    return np.repeat(low_shifts * scale_factor, scale_factor)[:omega_rows.shape[0]]


def build_disk_texture_rotating_state(n_phi: int = 1024, n_r: int = 512, seed: int = 42,
                                      r_inner: float = 2.0, r_outer: float = 3.5,
                                      enable_rt: bool = True, color_temp: float = None,
                                      generation_scale: int = 2) -> DiskTextureRotatingState:
    """预计算 `parametric` 旋转纹理的静态状态。"""
    generation_scale = _validate_disk_generation_scale(generation_scale)

    if color_temp is None:
        color_temp = DISK_COLOR_TEMPERATURE

    rng = np.random.default_rng(seed)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_norm = np.linspace(0, 1, n_r)
    phi_grid_base, r_norm_grid = np.meshgrid(phi, r_norm)

    r_vals = r_inner + (r_outer - r_inner) * r_norm_grid
    disk_area = (r_outer ** 2 - r_inner ** 2) / 10.0
    omega_grid = np.sqrt(0.5 / (r_vals ** 3 + 1e-6))

    temp_base = _generate_temperature_base(rng, n_r, n_phi, r_norm_grid)
    spiral, spiral_temp = _generate_spiral_arms(
        rng, n_r, n_phi, phi_grid_base, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )
    turbulence, kep_shift_pixels, turb_temp = _generate_turbulence(
        rng, n_r, n_phi, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )
    arcs, arcs_temp = _generate_filaments(
        rng, n_r, n_phi, phi_grid_base, r_norm_grid, disk_area, 0.0, None, generation_scale=generation_scale
    )
    rt_spikes, rt_temp = _generate_rt_spikes(
        rng, n_r, n_phi, phi_grid_base, r_norm_grid, disk_area, enable_rt, 0.0, None, generation_scale=generation_scale
    )
    hotspot, hotspot_temp = _generate_hotspots(rng, n_r, n_phi, phi_grid_base, r_norm_grid, disk_area, 0.0, None)
    az_hotspot = _generate_azimuthal_hotspot(
        rng, n_r, n_phi, phi_grid_base, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )
    disturb_mod = _generate_disturbance_mod(
        rng, n_r, n_phi, kep_shift_pixels, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )

    return DiskTextureRotatingState(
        n_phi=n_phi,
        n_r=n_r,
        seed=seed,
        generation_scale=generation_scale,
        r_inner=r_inner,
        r_outer=r_outer,
        enable_rt=enable_rt,
        color_temp=float(color_temp),
        omega_rows=omega_grid[:, 0].astype(np.float32),
        edge=compute_edge_alpha(n_r).astype(np.float32),
        temp_base=temp_base.astype(np.float32),
        spiral=spiral.astype(np.float32),
        spiral_temp=spiral_temp.astype(np.float32),
        turbulence=turbulence.astype(np.float32),
        turb_temp=turb_temp.astype(np.float32),
        arcs=arcs.astype(np.float32),
        arcs_temp=arcs_temp.astype(np.float32),
        rt_spikes=rt_spikes.astype(np.float32),
        rt_temp=rt_temp.astype(np.float32),
        hotspot=hotspot.astype(np.float32),
        hotspot_temp=hotspot_temp.astype(np.float32),
        az_hotspot=az_hotspot.astype(np.float32),
        disturb_mod=disturb_mod.astype(np.float32),
    )


def _generate_disk_texture_rotating_from_state(state: DiskTextureRotatingState,
                                               t_offset: float = 0.0,
                                               color_temp: float = None) -> np.ndarray:
    """基于预计算状态生成某一时刻的旋转纹理。"""
    if color_temp is None:
        color_temp = state.color_temp

    full_res_rot = _compute_rotation_pixels(state.omega_rows, t_offset, state.n_phi)
    low_res_rot = _compute_upscaled_rotation_pixels(
        state.omega_rows, t_offset, state.n_phi, scale_factor=state.generation_scale
    )

    temp_base = _roll_rows(state.temp_base, full_res_rot)
    spiral = _roll_rows(state.spiral, low_res_rot)
    spiral_temp = _roll_rows(state.spiral_temp, low_res_rot)
    turbulence = _roll_rows(state.turbulence, low_res_rot)
    turb_temp = _roll_rows(state.turb_temp, low_res_rot)
    arcs = _roll_rows(state.arcs, low_res_rot)
    arcs_temp = _roll_rows(state.arcs_temp, low_res_rot)
    rt_spikes = _roll_rows(state.rt_spikes, low_res_rot)
    rt_temp = _roll_rows(state.rt_temp, low_res_rot)
    hotspot = _roll_rows(state.hotspot, full_res_rot)
    hotspot_temp = _roll_rows(state.hotspot_temp, full_res_rot)
    az_hotspot = _roll_rows(state.az_hotspot, low_res_rot)
    disturb_mod = _roll_rows(state.disturb_mod, low_res_rot)

    temp_struct = spiral_temp + turb_temp + arcs_temp + rt_temp + hotspot_temp
    rt_weight = 0.20 if state.enable_rt else 0.0
    density = 0.15 + 0.10 * spiral + 0.15 * turbulence + 0.20 * hotspot + 0.30 * arcs + rt_weight * rt_spikes

    density = density * disturb_mod
    temp_struct = temp_struct * disturb_mod

    return _compose_disk_texture_from_fields(temp_base, temp_struct, density, az_hotspot, state.edge, color_temp)


def _tileable_noise(shape: Tuple[int, int], rng: np.random.Generator, freq_u: int = 6, freq_v: int = 6) -> np.ndarray:
    """用多条弧线生成云雾效果，保证 phi 方向无缝。"""
    h, w = shape

    cloud = np.zeros((h, w), dtype=np.float32)
    n_arcs = rng.integers(30, 60)

    for _ in range(n_arcs):
        arc_phi = rng.uniform(0, 2 * np.pi)
        arc_r = np.sqrt(rng.uniform(0.0, 1.0))
        arc_phi_width = rng.uniform(0.15, 0.5)
        arc_r_width = rng.uniform(0.03, 0.08)
        arc_intensity = rng.uniform(0.03, 0.12)

        kappa = 1.0 / (arc_phi_width ** 2) * 0.6

        phi = np.linspace(0, 2 * np.pi, w, endpoint=False)
        r_norm = np.linspace(0, 1, h)
        phi_grid, r_grid = np.meshgrid(phi, r_norm)

        r_diff = r_grid - arc_r
        arc_val = np.exp(kappa * (np.cos(phi_grid - arc_phi) - 1))
        arc_val *= np.exp(-0.5 * (r_diff / arc_r_width) ** 2)
        arc_val *= arc_intensity

        cloud += arc_val

    cloud = np.clip(cloud, 0, 1)
    return cloud


def _periodic_pixel_noise(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """生成像素级白噪声，保证 phi 方向周期性（首尾相接）。

    用于湍流的 pixel_noise 层，提供高频颗粒感，同时保证纹理无缝。
    """
    h, w = shape
    noise = rng.random((h, w)).astype(np.float32)
    noise[:, -1] = noise[:, 0]  # 强制周期性：phi=0 和 phi=2π 相同
    return noise * 2 - 1  # 返回 [-1, 1] 范围


def _fbm_noise(shape: Tuple[int, int], rng: np.random.Generator, octaves: int = 4, persistence: float = 0.5, base_scale: int = 1, wrap_u: bool = False) -> np.ndarray:
    """分形布朗运动噪声（多层叠加）。wrap_u=True 时用 tileable 噪声替代。"""
    if wrap_u:
        result = np.zeros(shape, dtype=np.float32)
        for i in range(octaves):
            freq = int(base_scale * (2 ** i))
            tile_noise = _tileable_noise(shape, rng, freq_u=max(2, freq), freq_v=max(1, freq // 2))
            result += tile_noise * (persistence ** i)
        result /= np.max(result) + 1e-6
        return result
    result = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for i in range(octaves):
        scale = base_scale * (2 ** i)
        sh = max(shape[0] // scale, 2)
        sw = max(shape[1] // scale, 2)
        small = rng.random((sh, sw)).astype(np.float32)
        pil = Image.fromarray((small * 255).astype(np.uint8))
        up = np.array(pil.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)) / 255.0
        result += up * amplitude
        total_amp += amplitude
        amplitude *= persistence
    return result / total_amp


def _blend_azimuthal_seam(tex: np.ndarray, seam_width: int = 64) -> np.ndarray:
    """
    将纹理在 u=0/u=2π 方向做平滑过渡，避免拼接时出现明显缝隙。
    """
    if seam_width <= 0:
        return tex
    if seam_width * 2 >= tex.shape[1]:
        return tex
    tex_blended = tex.copy()
    left = tex[:, :seam_width, :].copy()
    right = tex[:, -seam_width:, :].copy()
    for i in range(seam_width):
        t = (i + 1) / (seam_width + 1)
        tex_blended[:, i, :] = (1 - t) * left[:, i, :] + t * right[:, i, :]
        tex_blended[:, -seam_width + i, :] = (1 - t) * right[:, i, :] + t * left[:, i, :]
    return tex_blended


def generate_disk_mipmaps(base_tex: np.ndarray, levels: int = 4) -> np.ndarray:
    """生成吸积盘纹理的 mipmap 金字塔"""
    mips = [base_tex.copy()]
    for _ in range(levels):
        h, w = mips[-1].shape[:2]
        if h < 2 or w < 2:
            break
        new_h, new_w = h // 2, w // 2
        down = np.zeros((new_h, new_w, 4), dtype=np.float32)
        down = (mips[-1][0::2, 0::2] + mips[-1][1::2, 0::2] +
                mips[-1][0::2, 1::2] + mips[-1][1::2, 1::2]) / 4.0
        mips.append(down.astype(np.float32))
    return mips


def compute_disk_texture_resolution(width: int, height: int, cam_pos: List[float], fov: float, r_inner: float, r_outer: float, rs: float = 1.0) -> Tuple[int, int]:
    """
    根据相机参数计算吸积盘纹理分辨率。
    n_phi: 基于视角覆盖的角分辨率，每个像素约 1 个 phi 样本
    n_r: 基于径向覆盖的分辨率，每个径向单位约 0.5 个样本
    """
    camera_distance = math.sqrt(cam_pos[0]**2 + cam_pos[1]**2 + cam_pos[2]**2)

    disk_angular_radius = math.atan(r_outer / camera_distance)
    disk_angular_extent = 2 * disk_angular_radius
    screen_fraction = fov * math.pi / 180.0

    n_phi = int(width * (disk_angular_extent / screen_fraction))
    n_r = int(height * (disk_angular_radius / screen_fraction) * 0.5)

    n_phi = max(256, n_phi)
    n_r = max(128, n_r)

    n_phi = n_phi + (16 - n_phi % 16) % 16
    n_r = n_r + (16 - n_r % 16) % 16

    return n_phi, n_r


def load_cached_disk_texture(width: Optional[int] = None, height: Optional[int] = None, cam_pos: Optional[List[float]] = None, fov: Optional[float] = None,
                               seed: int = 42, r_inner: float = 2.0, r_outer: float = 3.5, force: bool = False,
                               generation_scale: int = 2) -> np.ndarray:
    """
    加载或生成吸积盘纹理（带缓存）。
    - width, height, cam_pos, fov: 用于计算纹理分辨率
    - seed: 随机种子
    - r_inner, r_outer: 吸积盘内外半径
    - force: 强制重新生成，忽略缓存
    返回 (n_r, n_phi, 4) float32
    """
    generation_scale = _validate_disk_generation_scale(generation_scale)

    if width and height and cam_pos and fov:
        n_phi, n_r = compute_disk_texture_resolution(width, height, cam_pos, fov, r_inner, r_outer)
    else:
        n_phi, n_r = 1024, 512

    cache_dir = "output/.disk_texture_cache"
    cache_key = f"disk_{r_inner:.2f}_{r_outer:.2f}_{seed}_{n_phi}x{n_r}_scale{generation_scale}.npy"
    cache_path = os.path.join(cache_dir, cache_key)

    if not force and os.path.exists(cache_path):
        print(f"Loading cached disk texture: {cache_key}")
        return np.load(cache_path)

    print(f"Generating disk texture: r_inner={r_inner}, r_outer={r_outer}, seed={seed}, n_phi={n_phi}, n_r={n_r}")
    tex = generate_disk_texture(
        n_phi=n_phi, n_r=n_r, seed=seed, r_inner=r_inner, r_outer=r_outer,
        generation_scale=generation_scale,
    )

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, tex)
    print(f"Cached to: {cache_path}")
    return tex


def _generate_spiral_arms(rng: np.random.Generator, n_r: int, n_phi: int, phi_grid: np.ndarray, r_norm_grid: np.ndarray,
                           t_offset: float = 0.0, omega_grid: np.ndarray = None,
                           generation_scale: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成螺旋臂密度和温度贡献
    返回：(spiral, temp_contribution)

    优化：使用 2x 低分辨率生成 + upscale，获得约 5x 加速比
    """
    if not ENABLE_DISK_SPIRAL_ARMS:
        zeros = np.zeros((n_r, n_phi), dtype=np.float32)
        return zeros, zeros

    # ===== 性能优化：2x 低分辨率生成 + upscale =====
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    # 从传入的网格降级采样（保留旋转信息）
    low_phi_grid = phi_grid[::scale_factor, ::scale_factor]
    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]

    n_arms = rng.integers(2, 5)
    n_from_center = rng.integers(2, 4)

    # 在低分辨率下生成
    low_spiral = np.zeros((low_n_r, low_n_phi), dtype=np.float32)
    low_temp_contribution = np.zeros((low_n_r, low_n_phi), dtype=np.float32)

    used_angles = []
    for arm_idx in tqdm(range(n_arms), desc="Spiral arms (2x lowres)", leave=False):
        if arm_idx < n_from_center:
            r_start = 0.0
            base_angle = arm_idx * 2 * np.pi / n_from_center
        else:
            r_start = rng.uniform(0.05, 0.5)
            base_angle = rng.uniform(0, 2 * np.pi)

        for existing in used_angles:
            if abs(base_angle - existing) < 0.4:
                base_angle = (base_angle + 0.5) % (2 * np.pi)
        used_angles.append(base_angle)

        rotations = rng.uniform(2.5, 5.0)
        base_width = rng.uniform(0.2, 0.4)
        arm_delta_T = rng.uniform(0.1, 0.3)

        r_length = rotations / 6.0 * (1.0 - r_start)
        r_length = min(r_length, 1.0 - r_start)

        # 每条螺旋臂由 4-8 个 sub-arm 段组成，每段之间有明显的间隙
        sub_arm_count = rng.integers(4, 9)
        sub_arm_fill = rng.uniform(0.4, 0.6)  # sub-arm 占总长度的比例（40-60%）
        sub_arm_lengths = rng.uniform(0.08, 0.20, sub_arm_count)
        sub_arm_lengths = sub_arm_lengths / sub_arm_lengths.sum() * r_length * sub_arm_fill

        # sub-arm 的起始径向位置 - 大间隙让分段更明显
        sub_r_starts = np.zeros(sub_arm_count)
        for j in range(1, sub_arm_count):
            gap = rng.uniform(0.08, 0.15)  # 大间隙
            sub_r_starts[j] = sub_r_starts[j-1] + sub_arm_lengths[j-1] + gap
        sub_r_starts += r_start

        # sub-arm 的宽度和强度变化 - 增加对比度
        sub_widths = base_width * rng.uniform(0.3, 2.5, sub_arm_count)
        sub_widths = np.clip(sub_widths, 0.06, 1.2)
        sub_intensities = rng.uniform(0.1, 0.7, sub_arm_count)

        # 预先生成 arm_noise（在 sub-arm 循环外，避免重复计算）
        arm_noise = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=3, freq_v=2)

        for j in range(sub_arm_count):
            sr = sub_r_starts[j]
            sr_len = sub_arm_lengths[j]
            sr_width = sub_widths[j]
            sr_int = sub_intensities[j]
            sr_end = sr + sr_len

            # 螺旋臂角度公式
            arm_angle = low_phi_grid - base_angle + low_r_norm_grid * rotations * 2 * np.pi

            # 宽度调制
            width_mod = 0.2 + 1.5 * arm_noise
            width_mod = np.clip(width_mod, 0.15, 3.0)

            arm_kappa = 1.0 / (sr_width ** 2) * 1.5
            arm_val = np.exp(arm_kappa * (np.cos(arm_angle) - 1) * width_mod)

            # 径向 mask - 使用硬边界，减少 fade 效果
            mask = (low_r_norm_grid >= sr) & (low_r_norm_grid <= sr_end)
            arm_val = np.where(mask, arm_val, 0)

            # 强度调制 - 降低断裂效果，让 sub-arm 更连续
            intensity_mod = 0.1 + 0.9 * (arm_noise ** 0.15)

            # 轻微的边缘软化（比之前小）
            fade_edge = 0.02
            fade_in = np.clip((low_r_norm_grid - sr) / fade_edge, 0, 1)
            fade_out = np.clip((sr_end - low_r_norm_grid) / fade_edge, 0, 1)
            arm_val *= fade_in * fade_out * sr_int * intensity_mod

            low_spiral += arm_val
            low_temp_contribution += arm_val * arm_delta_T

    low_spiral = np.clip(low_spiral / (np.max(low_spiral) + 1e-6), 0, 1)

    # 使用 np.kron 进行 upscale
    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    spiral = np.kron(low_spiral, upscale_kernel)
    temp_contribution = np.kron(low_temp_contribution, upscale_kernel)

    # 裁剪到目标尺寸（防止整除时尺寸不匹配）
    spiral = spiral[:n_r, :n_phi]
    temp_contribution = temp_contribution[:n_r, :n_phi]

    return spiral, temp_contribution


def _generate_turbulence(rng: np.random.Generator, n_r: int, n_phi: int, r_norm_grid: np.ndarray,
                          t_offset: float = 0.0, omega_grid: np.ndarray = None,
                          generation_scale: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成云雾/湍流密度和温度贡献
    返回：(turbulence, kep_shift_pixels, temp_contribution)

    优化：使用 2x 低分辨率生成 + upscale，获得约 2-3x 加速比
    """
    # ===== 性能优化：2x 低分辨率生成 + upscale =====
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    # 从传入的 r_norm_grid 降级采样
    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]

    shear_strength = rng.uniform(3.0, 6.0)
    # 低分辨率下的开普勒剪切
    kep_shear_low = shear_strength * (1.0 / (low_r_norm_grid + 0.3) ** 1.5 - 0.8)
    kep_shear_low = np.clip(kep_shear_low, 0, shear_strength * 8)
    kep_shift_pixels_low = (kep_shear_low / (2 * np.pi) * low_n_phi).astype(int)
    max_shift_low = low_n_phi // 4
    kep_shift_pixels_low = np.clip(kep_shift_pixels_low, -max_shift_low, max_shift_low)

    # 在低分辨率下生成 5 层噪声
    turbulence_coarse = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=8, freq_v=4)
    turbulence_mid = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=24, freq_v=12)
    turbulence_fine = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=80, freq_v=40)
    turbulence_extra = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=200, freq_v=100)
    turbulence_ultra = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=400, freq_v=200)

    # 应用开普勒剪切滚动（低分辨率）
    for layer in [turbulence_coarse, turbulence_mid, turbulence_fine, turbulence_extra, turbulence_ultra]:
        for ri in range(low_n_r):
            layer[ri, :] = np.roll(layer[ri, :], kep_shift_pixels_low[ri, 0])

    # 动态旋转支持（低分辨率）
    rotation_pixels_low = None
    if t_offset != 0.0 and omega_grid is not None:
        # 降级采样 omega_grid 到低分辨率
        omega_grid_low = omega_grid[::scale_factor, ::scale_factor]
        rotation_pixels_low = (t_offset * omega_grid_low / (2 * np.pi) * low_n_phi).astype(int)
        for layer in [turbulence_coarse, turbulence_mid, turbulence_fine, turbulence_extra, turbulence_ultra]:
            for ri in range(low_n_r):
                layer[ri, :] = np.roll(layer[ri, :], -rotation_pixels_low[ri, 0])

    # 像素级高频噪声（低分辨率）
    pixel_noise = _periodic_pixel_noise((low_n_r, low_n_phi), rng)

    # 对 pixel_noise 应用开普勒旋转（t_offset != 0 时）
    if rotation_pixels_low is not None:
        for ri in range(low_n_r):
            pixel_noise[ri, :] = np.roll(pixel_noise[ri, :], -rotation_pixels_low[ri, 0])

    # 湍流权重：多层噪声叠加
    turbulence_low = (0.08 * turbulence_coarse + 0.15 * turbulence_mid
                      + 0.25 * turbulence_fine + 0.22 * turbulence_extra
                      + 0.18 * turbulence_ultra + 0.12 * np.clip(pixel_noise, 0, 1))

    # upscale
    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    turbulence = np.kron(turbulence_low, upscale_kernel)[:n_r, :n_phi]

    temp_contribution = 0.05 * np.clip(turbulence, 0, 1)

    # 高分辨率 kep_shift_pixels（用于后续处理）
    kep_shear = shear_strength * (1.0 / (r_norm_grid + 0.3) ** 1.5 - 0.8)
    kep_shear = np.clip(kep_shear, 0, shear_strength * 8)
    kep_shift_pixels = (kep_shear / (2 * np.pi) * n_phi).astype(int)
    max_shift = n_phi // 4
    kep_shift_pixels = np.clip(kep_shift_pixels, -max_shift, max_shift)

    return turbulence, kep_shift_pixels, temp_contribution


def _generate_filaments(rng: np.random.Generator, n_r: int, n_phi: int, phi_grid: np.ndarray, r_norm_grid: np.ndarray, disk_area: float,
                       t_offset: float = 0.0, omega_grid: np.ndarray = None,
                       generation_scale: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成细丝（filaments）密度和温度贡献
    物理意义：吸积盘中的丝状结构，可能是磁重联或剪切流形成的细长条纹
    特征：沿角度方向延伸（长条状），径向很窄，由多个 sub-filament 接续而成

    数量说明：真实吸积盘中细丝结构约 20-50 条，这里用 30-60 条保证可见性

    优化：使用 2x 低分辨率生成 + upscale，获得约 5x 加速比
    """
    # ===== 性能优化：2x 低分辨率生成 + upscale =====
    # 在低分辨率下生成 filaments，然后 upscale 到目标分辨率
    # 测试表明：2x 低分辨率 + upscale 可获得 5x 加速，MSE 仅 0.044
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    # 从传入的网格降级采样（保留旋转信息）
    low_phi_grid = phi_grid[::scale_factor, ::scale_factor]
    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]

    # 细丝数量：150-300 条（每条由多个 sub-filament 组成）
    arc_count = int(rng.uniform(150, 300))
    # 每条细丝的 sub-filament 数量：2-4 个
    sub_filament_counts = rng.integers(2, 5, arc_count)

    arc_phi_starts = rng.uniform(0, 2 * np.pi, arc_count)
    r_positions = rng.uniform(0.05, 0.95, arc_count)
    arc_rs = 0.05 + r_positions ** 0.6 * 0.9
    # 细丝径向宽度：0.002-0.008（适中宽度）
    arc_r_widths = rng.uniform(0.002, 0.008, arc_count)
    # 细丝总角度长度：0.5-1.2（约 180°-430°）
    arc_lengths = rng.uniform(0.5, 1.2, arc_count)

    arc_intensities = rng.uniform(0.7, 1.0, arc_count)  # 提高细丝强度
    arc_delta_Ts = 0.3 + 0.6 * rng.power(0.3, arc_count)  # 提高温度贡献范围：0.3-0.9

    print(f"Generating {arc_count} filaments with sub-segments (2x lowres + upscale)...")

    # 在低分辨率下生成
    low_arcs = np.zeros((low_n_r, low_n_phi), dtype=np.float32)
    low_temp_contribution = np.zeros((low_n_r, low_n_phi), dtype=np.float32)

    # 逐条生成细丝，每条细丝由多个 sub-filament 接续而成
    for i in tqdm(range(arc_count), desc="Filaments", leave=False):
        # 细丝基础参数
        base_phi = arc_phi_starts[i]
        base_r = arc_rs[i]
        base_width = arc_r_widths[i]
        total_length = arc_lengths[i]
        intensity = arc_intensities[i]
        delta_T = arc_delta_Ts[i]

        # 生成 sub-filament 参数
        sub_count = sub_filament_counts[i]
        sub_fill = rng.uniform(0.35, 0.55)  # sub-filament 占总长度的比例
        sub_lengths = rng.uniform(0.08, 0.20, sub_count)
        # 归一化 sub 长度，使其总和等于 total_length * sub_fill
        sub_lengths = sub_lengths / sub_lengths.sum() * total_length * sub_fill

        # sub-filament 的起始角度（沿细丝方向分布）- 大间隙
        sub_starts = np.zeros(sub_count)
        sub_starts[0] = base_phi
        for j in range(1, sub_count):
            gap = rng.uniform(0.08, 0.20)  # 大间隙
            sub_starts[j] = sub_starts[j-1] + sub_lengths[j-1] + gap

        # sub-filament 的宽度和强度变化 - 增加对比度
        sub_widths = base_width * rng.uniform(0.3, 3.0, sub_count)
        sub_widths = np.clip(sub_widths, 0.001, 0.025)
        sub_intensities = intensity * rng.uniform(0.15, 1.0, sub_count)

        # 生成每个 sub-filament
        for j in range(sub_count):
            sub_phi = sub_starts[j]
            sub_len = sub_lengths[j]
            sub_w = sub_widths[j]
            sub_int = sub_intensities[j]

            # 角度剖面
            phi_range = sub_len / (base_r + 0.01)
            phi_half_width = np.maximum(phi_range * 0.7, 0.2)
            kappa = 1.5 / (phi_half_width ** 2)

            sub_val = np.exp(kappa * (np.cos(low_phi_grid - sub_phi) - 1))

            # 径向剖面
            r_diff = low_r_norm_grid - base_r
            r_prof = np.exp(-0.5 * (r_diff / sub_w) ** 2)

            low_arcs += sub_val * r_prof * sub_int
            low_temp_contribution += sub_val * r_prof * sub_int * delta_T * 0.7

    # 使用 np.kron 进行 upscale
    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    arcs = np.kron(low_arcs, upscale_kernel)
    temp_contribution = np.kron(low_temp_contribution, upscale_kernel)

    # 裁剪到目标尺寸（防止整除时尺寸不匹配）
    arcs = arcs[:n_r, :n_phi]
    temp_contribution = temp_contribution[:n_r, :n_phi]

    arcs = np.clip(arcs, 0, 1)
    temp_contribution = np.clip(temp_contribution, 0, arcs * 0.5)
    return arcs, temp_contribution


def _generate_rt_spikes(rng: np.random.Generator, n_r: int, n_phi: int, phi_grid: np.ndarray, r_norm_grid: np.ndarray, disk_area: float, enable_rt: bool,
                       t_offset: float = 0.0, omega_grid: np.ndarray = None,
                       generation_scale: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 Rayleigh-Taylor 不稳定性密度和温度贡献
    返回：(rt_spikes, temp_contribution)

    优化：使用 2x 低分辨率生成 + upscale，获得约 4-5x 加速比
    """
    # 如果禁用 RT，返回零数组
    if not enable_rt:
        return np.zeros((n_r, n_phi), dtype=np.float32), np.zeros((n_r, n_phi), dtype=np.float32)

    # ===== 性能优化：2x 低分辨率生成 + upscale =====
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    # 从传入的网格降级采样（保留旋转信息）
    low_phi_grid = phi_grid[::scale_factor, ::scale_factor]
    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]

    # RT 不稳定性主要出现在内圈，数量增加
    rt_count = int(rng.uniform(15, 30) * disk_area * 0.8)

    rt_phis = rng.uniform(0, 2 * np.pi, rt_count)
    # RT 位置偏内圈，更多集中在 r_norm < 0.3 区域 - 使用幂次分布偏向内圈
    rt_r_bases = np.power(rng.uniform(0.01, 0.15, rt_count), 1.5)  # 幂次分布偏向内圈
    rt_phi_widths = rng.uniform(0.08, 0.20, rt_count)  # 更窄，更集中
    rt_r_lengths = rng.uniform(0.08, 0.20, rt_count)  # 更长
    rt_intensities = rng.uniform(0.8, 1.0, rt_count)  # 提高强度
    rt_delta_Ts = rng.uniform(0.5, 1.2, rt_count)  # 提高温度贡献

    # 在低分辨率下生成
    rt_spikes = np.zeros((low_n_r, low_n_phi), dtype=np.float32)
    temp_contribution = np.zeros((low_n_r, low_n_phi), dtype=np.float32)

    for i in range(rt_count):
        rt_phi_kappa = 1.0 / (rt_phi_widths[i] ** 2) * 1.5
        rt_val = np.exp(rt_phi_kappa * (np.cos(low_phi_grid - rt_phis[i]) - 1))

        rt_r_diff = low_r_norm_grid - rt_r_bases[i]
        r_fade_out = np.clip(rt_r_lengths[i] * 2 - rt_r_diff, 0, 1)
        r_fade_in = np.clip((low_r_norm_grid - rt_r_bases[i]) / (rt_r_lengths[i] * 0.3), 0, 1)
        rt_r_profile = np.exp(-0.5 * (rt_r_diff / (rt_r_lengths[i] * 0.4)) ** 2) * r_fade_out * r_fade_in

        rt_val *= rt_r_profile * rt_intensities[i]
        rt_spikes += rt_val
        temp_contribution += rt_val * rt_delta_Ts[i]

    rt_spikes = np.clip(rt_spikes, 0, 1)

    # 使用 np.kron 进行 upscale
    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    rt_spikes = np.kron(rt_spikes, upscale_kernel)[:n_r, :n_phi]
    temp_contribution = np.kron(temp_contribution, upscale_kernel)[:n_r, :n_phi]

    return rt_spikes, temp_contribution


def _generate_azimuthal_hotspot(rng: np.random.Generator, n_r: int, n_phi: int, phi_grid: np.ndarray, r_norm_grid: np.ndarray,
                                 t_offset: float = 0.0, omega_grid: np.ndarray = None,
                                 generation_scale: int = 2) -> np.ndarray:
    """
    生成方位热点（低频正弦 + 噪声，自转流动感）
    返回：az_hotspot

    优化：使用 2x 低分辨率生成 + upscale，获得约 2-3x 加速比
    """
    # ===== 性能优化：2x 低分辨率生成 + upscale =====
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    # 从传入的网格降级采样（保留旋转信息）
    low_phi_grid = phi_grid[::scale_factor, ::scale_factor]
    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]

    az_freq = rng.integers(2, 5)
    shear = low_r_norm_grid ** 1.2 * rng.uniform(2.0, 4.0)
    az_wave = 0.5 + 0.5 * np.sin((low_phi_grid + shear) * az_freq)
    az_noise = _fbm_noise((low_n_r, low_n_phi), rng, octaves=3, persistence=0.5, base_scale=3, wrap_u=True)

    # 对 az_noise 应用开普勒旋转（t_offset != 0 时）
    if t_offset != 0.0 and omega_grid is not None:
        omega_grid_low = omega_grid[::scale_factor, ::scale_factor]
        rotation_pixels_low = (t_offset * omega_grid_low / (2 * np.pi) * low_n_phi).astype(int)
        for ri in range(low_n_r):
            az_noise[ri, :] = np.roll(az_noise[ri, :], -rotation_pixels_low[ri, 0])

    az_hotspot_low = az_wave * az_noise

    # upscale
    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    az_hotspot = np.kron(az_hotspot_low, upscale_kernel)[:n_r, :n_phi]
    return az_hotspot


def _apply_disturbance(rng: np.random.Generator, n_r: int, n_phi: int, density: np.ndarray,
                        temp_struct: np.ndarray, kep_shift_pixels: np.ndarray,
                        r_norm_grid: np.ndarray, t_offset: float = 0.0,
                        omega_grid: np.ndarray = None, generation_scale: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply turbulence disturbance to density and temperature fields.
    Returns: (density, temp_struct)

    Args:
        t_offset: 时间偏移，用于动态旋转
        omega_grid: 开普勒角速度网格，用于计算旋转量

    优化：使用 2x 低分辨率生成 + upscale，获得约 1.5-2x 加速比
    """
    disturb_mod = _generate_disturbance_mod(
        rng, n_r, n_phi, kep_shift_pixels, r_norm_grid, t_offset, omega_grid,
        generation_scale=generation_scale,
    )
    density = density * disturb_mod
    temp_struct = temp_struct * disturb_mod
    return density, temp_struct


def _generate_hotspots(rng: np.random.Generator, n_r: int, n_phi: int, phi_grid: np.ndarray, r_norm_grid: np.ndarray, disk_area: float,
                      t_offset: float = 0.0, omega_grid: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成温度热点密度和温度贡献
    物理意义：吸积盘中的局部高温区域（如磁重联、激波碰撞形成的亮斑）
    特征：近似圆形或椭圆形的斑点，径向和角度宽度相近

    数量说明：真实吸积盘中观测到的热点约数个 - 数十个，这里用 20-40 个保证可见性
    """
    # 热点数量：20-40 个（物理合理范围）
    hotspot_count = int(rng.uniform(20, 40))
    hotspot_delta_Ts = 0.5 + 2.5 * rng.power(0.4, hotspot_count)

    h_phis = rng.uniform(0, 2 * np.pi, hotspot_count)
    r_rands = rng.uniform(0, 1, hotspot_count)
    h_rs = 0.1 + r_rands ** 0.6 * 0.85
    # 热点角度宽度：0.08-0.20（约 30°-70°），较宽形成斑点
    h_phi_widths = rng.uniform(0.08, 0.20, hotspot_count)
    # 热点径向宽度：0.02-0.05，与角度宽度相近，形成近似圆形的斑点
    h_r_widths = 0.02 + rng.uniform(0, 0.03, hotspot_count)
    h_intensities = 0.3 + (1 - h_rs) * 0.6 + rng.uniform(0, 0.1, hotspot_count)

    print(f"Generating {hotspot_count} hotspots...")
    hotspot = np.zeros((n_r, n_phi), dtype=np.float32)
    batch_size = 400

    for batch_start in tqdm(range(0, hotspot_count, batch_size), desc="Hotspots", leave=False):
        batch_end = min(batch_start + batch_size, hotspot_count)

        h_ps = h_phis[batch_start:batch_end, None, None]
        hs = h_rs[batch_start:batch_end, None, None]
        hp_ws = h_phi_widths[batch_start:batch_end, None, None]
        hr_ws = h_r_widths[batch_start:batch_end, None, None]
        h_ints = h_intensities[batch_start:batch_end, None, None]

        kappa = 1.0 / (hp_ws ** 2) * 1.5
        h_batch = np.exp(kappa * (np.cos(phi_grid[None, :, :] - h_ps) - 1.0))
        r_diff = r_norm_grid[None, :, :] - hs
        h_batch *= np.exp(-0.5 * (r_diff / hr_ws) ** 2)
        h_batch *= h_ints

        hotspot += np.sum(h_batch, axis=0)

    hotspot = np.clip(hotspot, 0, 1)
    temp_contribution = 0.12 * hotspot
    return hotspot, temp_contribution


def _spawn_single_filament(rng: np.random.Generator, n_r: int, n_phi: int,
                           r_norm_all: np.ndarray, omega_all: np.ndarray
                           ) -> tuple:
    """Generate a single filament blob for entity lifecycle system.

    Creates a circular 2D Gaussian blob that will be naturally sheared into
    an arc by differential Keplerian rotation during rendering.

    Args:
        rng: numpy random generator
        n_r: total number of radial rows
        n_phi: number of azimuthal columns
        r_norm_all: normalized radial position for each row, shape (n_r,)
        omega_all: Keplerian angular velocity for each row, shape (n_r,)

    Returns:
        (row_indices, phi_density, phi_temp, omega, source_phi, total_extent,
         sigma_r, sigma_phi0, peak_density, peak_temp, base_r):
        - row_indices: int array of affected row indices, shape (n_affected,)
        - phi_density: empty array (blob computed at render time)
        - phi_temp: empty array (blob computed at render time)
        - omega: Keplerian angular velocity at entity center (rad/s)
        - source_phi: azimuthal center of the blob (rad)
        - total_extent: set to 2*pi (blob can stretch to any length)
        - sigma_r: radial Gaussian width (r_norm units)
        - sigma_phi0: initial azimuthal Gaussian width (rad)
        - peak_density: initial density peak value
        - peak_temp: initial temperature peak value
        - base_r: radial center in r_norm units

    Physical Meaning:
        Represents a magnetic reconnection event that creates a hot, compact
        blob. The blob is initially circular in (r, phi) space. Differential
        Keplerian rotation + turbulent stretching naturally deform it into
        a thin, elongated arc over time.
    """
    source_phi = float(rng.uniform(0, 2 * np.pi))
    r_pos = float(rng.uniform(0.05, 0.95))
    base_r = 0.05 + r_pos ** 0.6 * 0.9
    sigma_r = float(rng.uniform(0.005, 0.015))
    sigma_phi0 = float(rng.uniform(0.04, 0.10))
    peak_density = float(rng.uniform(0.5, 1.0))
    peak_temp = peak_density * float(rng.uniform(0.15, 0.35))

    row_mask = np.abs(r_norm_all - base_r) < 4 * sigma_r
    row_indices = np.where(row_mask)[0]
    if len(row_indices) == 0:
        center_idx = int(np.argmin(np.abs(r_norm_all - base_r)))
        row_indices = np.array([center_idx])

    center_idx = int(np.argmin(np.abs(r_norm_all - base_r)))
    omega = float(omega_all[center_idx])

    return (row_indices, np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32), omega, source_phi, 2 * np.pi,
            sigma_r, sigma_phi0, peak_density, peak_temp, base_r)


def _spawn_single_hotspot(rng: np.random.Generator, n_r: int, n_phi: int,
                          r_norm_all: np.ndarray, omega_all: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a single hotspot instance for entity lifecycle system.

    Creates one hotspot — an approximately circular bright patch. Parameters
    match _generate_hotspots statistics.

    Args:
        rng: numpy random generator
        n_r: total number of radial rows
        n_phi: number of azimuthal columns
        r_norm_all: normalized radial position for each row, shape (n_r,)
        omega_all: Keplerian angular velocity for each row, shape (n_r,)

    Returns:
        (row_indices, phi_density, phi_temp, omega):
        - row_indices: int array of affected row indices, shape (n_affected,)
        - phi_density: density contribution, shape (n_affected, n_phi)
        - phi_temp: temperature contribution, shape (n_affected, n_phi)
        - omega: Keplerian angular velocity at entity center (rad/s)

    Physical Meaning:
        Represents a localized high-temperature region in the accretion disk,
        caused by magnetic reconnection or shock collisions. Approximately
        circular in shape, with both radial and azimuthal Gaussian profiles.
    """
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    h_phi = float(rng.uniform(0, 2 * np.pi))
    r_rand = float(rng.uniform(0, 1))
    h_r = 0.1 + r_rand ** 0.6 * 0.85
    h_phi_width = float(rng.uniform(0.08, 0.20))
    h_r_width = 0.02 + float(rng.uniform(0, 0.03))
    h_intensity = 0.3 + (1 - h_r) * 0.6 + float(rng.uniform(0, 0.1))
    h_delta_T = 0.5 + 2.5 * float(rng.power(0.4))

    # 受影响行（3 sigma 截断）
    r_min = h_r - 3 * h_r_width
    r_max = h_r + 3 * h_r_width
    row_mask = (r_norm_all >= r_min) & (r_norm_all <= r_max)
    row_indices = np.where(row_mask)[0]

    if len(row_indices) == 0:
        center_idx = int(np.argmin(np.abs(r_norm_all - h_r)))
        row_indices = np.array([center_idx])

    r_subset = r_norm_all[row_indices]
    n_rows = len(row_indices)

    kappa = 1.5 / (h_phi_width ** 2)
    phi_prof = np.exp(kappa * (np.cos(phi - h_phi) - 1))

    phi_density = np.zeros((n_rows, n_phi), dtype=np.float32)
    phi_temp = np.zeros((n_rows, n_phi), dtype=np.float32)

    for k in range(n_rows):
        r_diff = r_subset[k] - h_r
        r_prof = np.exp(-0.5 * (r_diff / (h_r_width + 1e-8)) ** 2)
        phi_density[k] = phi_prof * r_prof * h_intensity
        phi_temp[k] = phi_density[k] * 0.12

    phi_density = np.clip(phi_density, 0, 1)
    phi_temp = np.clip(phi_temp, 0, 1)

    center_idx = int(np.argmin(np.abs(r_norm_all - h_r)))
    omega = float(omega_all[center_idx])

    return row_indices, phi_density, phi_temp, omega


def _spawn_single_rt_spike(rng: np.random.Generator, n_r: int, n_phi: int,
                           r_norm_all: np.ndarray, omega_all: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a single RT spike instance for entity lifecycle system.

    Creates one Rayleigh-Taylor instability spike. Parameters match
    _generate_rt_spikes statistics. RT spikes are biased toward the inner disk.

    Args:
        rng: numpy random generator
        n_r: total number of radial rows
        n_phi: number of azimuthal columns
        r_norm_all: normalized radial position for each row, shape (n_r,)
        omega_all: Keplerian angular velocity for each row, shape (n_r,)

    Returns:
        (row_indices, phi_density, phi_temp, omega):
        - row_indices: int array of affected row indices, shape (n_affected,)
        - phi_density: density contribution, shape (n_affected, n_phi)
        - phi_temp: temperature contribution, shape (n_affected, n_phi)
        - omega: Keplerian angular velocity at entity center (rad/s)

    Physical Meaning:
        Represents a Rayleigh-Taylor instability — a radial finger-like structure
        near the inner disk edge, where denser outer material plunges inward.
        Biased toward small r_norm (inner disk) with power-law distribution.
    """
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    rt_phi = float(rng.uniform(0, 2 * np.pi))
    rt_r_base = float(np.power(rng.uniform(0.01, 0.15), 1.5))
    rt_phi_width = float(rng.uniform(0.08, 0.20))
    rt_r_length = float(rng.uniform(0.08, 0.20))
    rt_intensity = float(rng.uniform(0.8, 1.0))
    rt_delta_T = float(rng.uniform(0.5, 1.2))

    # 受影响行：从 r_base 向外延伸 2*r_length
    r_min = max(rt_r_base - 0.02, 0.0)
    r_max = rt_r_base + rt_r_length * 2.5
    row_mask = (r_norm_all >= r_min) & (r_norm_all <= r_max)
    row_indices = np.where(row_mask)[0]

    if len(row_indices) == 0:
        center_idx = int(np.argmin(np.abs(r_norm_all - rt_r_base)))
        row_indices = np.array([center_idx])

    r_subset = r_norm_all[row_indices]
    n_rows = len(row_indices)

    rt_phi_kappa = 1.5 / (rt_phi_width ** 2)
    phi_prof = np.exp(rt_phi_kappa * (np.cos(phi - rt_phi) - 1))

    phi_density = np.zeros((n_rows, n_phi), dtype=np.float32)
    phi_temp = np.zeros((n_rows, n_phi), dtype=np.float32)

    for k in range(n_rows):
        rt_r_diff = r_subset[k] - rt_r_base
        r_fade_out = np.clip(rt_r_length * 2 - rt_r_diff, 0, 1)
        r_fade_in = np.clip((r_subset[k] - rt_r_base) / (rt_r_length * 0.3 + 1e-8), 0, 1)
        rt_r_prof = (np.exp(-0.5 * (rt_r_diff / (rt_r_length * 0.4 + 1e-8)) ** 2)
                     * r_fade_out * r_fade_in)
        phi_density[k] = phi_prof * rt_r_prof * rt_intensity
        phi_temp[k] = phi_density[k] * rt_delta_T

    phi_density = np.clip(phi_density, 0, 1)

    center_r = rt_r_base + rt_r_length * 0.5
    center_idx = int(np.argmin(np.abs(r_norm_all - center_r)))
    omega = float(omega_all[center_idx])

    return row_indices, phi_density, phi_temp, omega


def generate_disk_texture(n_phi: int = 1024, n_r: int = 512, seed: int = 42,
                          r_inner: float = 2.0, r_outer: float = 3.5,
                          enable_rt: bool = True, color_temp: float = None,
                          generation_scale: int = 2) -> np.ndarray:
    """
    直接在极坐标下生成吸积盘纹理，避免笛卡尔到极坐标的映射接缝问题。

    Args:
        n_phi: 角度方向分辨率（对应 0-2π）
        n_r: 径向方向分辨率（对应 r_inner 到 r_outer）
        seed: 随机种子
        r_inner: 内半径
        r_outer: 外半径
        enable_rt: 是否启用 Rayleigh-Taylor 不稳定性
        color_temp: 色温（单位：K），控制整体颜色。默认 None 使用 DISK_COLOR_TEMPERATURE

    Returns:
        (n_r, n_phi, 4) float32，第 4 通道为 alpha（面密度）
    """
    # 使用全局色温参数或传入的色温
    if color_temp is None:
        color_temp = DISK_COLOR_TEMPERATURE

    rng = np.random.default_rng(seed)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_norm = np.linspace(0, 1, n_r)
    phi_grid, r_norm_grid = np.meshgrid(phi, r_norm)

    r_vals = r_inner + (r_outer - r_inner) * r_norm_grid
    disk_area = (r_outer ** 2 - r_inner ** 2) / 10.0

    # ----- 温度基底（内热外冷 + 噪声扰动）-----
    # 先生成径向递减的基底，再叠加轻微噪声，最终数值控制在 0~0.45
    radial_decay = np.clip(1.0 - r_norm_grid, 0, 1) ** 1.3
    temp_coarse = _fbm_noise((n_r, n_phi), rng, octaves=4, persistence=0.6, base_scale=8, wrap_u=True)
    temp_fine = _fbm_noise((n_r, n_phi), rng, octaves=5, persistence=0.45, base_scale=3, wrap_u=True)
    temp_noise = 0.6 * temp_coarse + 0.4 * temp_fine
    temp_base = np.clip(radial_decay * (0.85 + 0.15 * temp_noise), 0, 1)
    temp_base *= 0.25

    # 各结构的温度贡献将在下面的循环中累积
    temp_struct = np.zeros((n_r, n_phi), dtype=np.float32)

    # ----- 密度场 -----
# 1) 螺旋臂
    generation_scale = _validate_disk_generation_scale(generation_scale)
    spiral, spiral_temp = _generate_spiral_arms(
        rng, n_r, n_phi, phi_grid, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )
    temp_struct += spiral_temp


    # 2) 云雾
    turbulence, kep_shift_pixels, turb_temp = _generate_turbulence(
        rng, n_r, n_phi, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )
    temp_struct += turb_temp

    # 3) Filaments
    arcs, arcs_temp = _generate_filaments(
        rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, 0.0, None, generation_scale=generation_scale
    )
    temp_struct += arcs_temp

    # 4) Rayleigh-Taylor 不稳定性
    rt_spikes, rt_temp = _generate_rt_spikes(
        rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, enable_rt, 0.0, None, generation_scale=generation_scale
    )
    temp_struct += rt_temp

    # 5) 温度热点
    hotspot, hotspot_temp = _generate_hotspots(rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, 0.0, None)
    temp_struct += hotspot_temp

    # 5) 方位热点
    az_hotspot = _generate_azimuthal_hotspot(
        rng, n_r, n_phi, phi_grid, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )

    # 组合密度
    rt_weight = 0.20 if enable_rt else 0.0
    density = 0.15 + 0.10 * spiral + 0.15 * turbulence + 0.20 * hotspot + 0.30 * arcs + rt_weight * rt_spikes

# 湍流扰动 - 降低 disturbance 强度，保留更多 spiral arm 和 filament 的分段结构
    density, temp_struct = _apply_disturbance(
        rng, n_r, n_phi, density, temp_struct, kep_shift_pixels, r_norm_grid, 0.0, None,
        generation_scale=generation_scale,
    )

    # 边缘软化（沿径向）
    edge = compute_edge_alpha(n_r)
    density *= edge[:, None]

    # 归一化
    density = np.clip(density / (np.percentile(density, 98) + 1e-6), 0, 1)

    # ----- 合成温度场（取 max，非相加）-----
    # temp_struct 先按 95 分位缩放到 0~1，再与基底比较
    if np.any(temp_struct > 0):
        struct_scale = np.percentile(temp_struct[temp_struct > 0], 95)
        temp_struct_scaled = temp_struct / (struct_scale + 1e-6)
    else:
        temp_struct_scaled = temp_struct
    temp_struct_scaled = np.clip(temp_struct_scaled * 0.8, 0, 1.2)

    # 基底在每个半径上不得高于该半径的结构温度（使用 P70 作为典型上限）
    struct_max_per_r = np.max(temp_struct_scaled, axis=1)
    struct_p70_per_r = np.quantile(temp_struct_scaled, 0.7, axis=1)
    struct_ceiling = np.maximum(struct_p70_per_r, 0.05)
    temp_base = np.minimum(temp_base, struct_ceiling[:, None])
    temp_base = np.minimum(temp_base, struct_max_per_r[:, None])

    temperature_field = np.clip(np.maximum(temp_base, temp_struct_scaled), 0, 1)

    # ----- 颜色（温度 -> 黑体辐射 RGB）-----
    # 色温控制温度映射范围：
    # - 2700K: 整体温度降低，更多区域处于红橙温度 (1500K-6000K)
    # - 4500K: 中等温度范围 (2000K-9000K)
    # - 6500K: 整体温度升高，更多区域处于白色温度 (3000K-12000K)
    # 使用线性插值：以 4500K 为基准，色温变化时调整 T_min 和 T_max
    t_factor = (color_temp - 4500) / (6500 - 2700)  # -0.47 ~ 0.47
    T_min = 2000 + t_factor * 1000  # 1500K ~ 2500K
    T_max = 9000 + t_factor * 3000  # 6000K ~ 12000K

    temp_aniso = np.clip(temperature_field * (0.9 + 0.25 * az_hotspot), 0, 1)
    T_K = T_min + temp_aniso * (T_max - T_min)
    bb_color = _blackbody_rgb(T_K)  # (n_r, n_phi, 3)
    # 高温端钳制：确保 R >= B，避免蓝色偏移（真正的白热不偏蓝）
    bb_color[:, :, 2] = np.minimum(bb_color[:, :, 2], bb_color[:, :, 0])

    # RGB = 黑体色 × 亮度（温度驱动），alpha = 密度（不透明度）
    # 亮度用 sqrt 而非 T^2/T^4，保留低温区可见的红/橙色
    luminosity = np.clip(np.sqrt(temp_aniso), 0, 1)

    tex = np.zeros((n_r, n_phi, 4), dtype=np.float32)
    tex[:, :, 0] = np.clip(bb_color[:, :, 0] * luminosity, 0, 1)
    tex[:, :, 1] = np.clip(bb_color[:, :, 1] * luminosity, 0, 1)
    tex[:, :, 2] = np.clip(bb_color[:, :, 2] * luminosity, 0, 1)
    tex[:, :, 3] = np.clip(density, 0, 1)  # alpha 只由面密度决定

    return tex


def generate_disk_texture_rotating(n_phi: int = 1024, n_r: int = 512, seed: int = 42,
                                    r_inner: float = 2.0, r_outer: float = 3.5,
                                    enable_rt: bool = True, t_offset: float = 0.0,
                                    color_temp: float = None,
                                    state: Optional[DiskTextureRotatingState] = None,
                                    generation_scale: int = 2) -> np.ndarray:
    """
    生成吸积盘纹理，支持参数化旋转（用于动画渲染）

    关键特性：
    - 使用固定的随机种子生成结构参数
    - 对温度基底和各组件应用开普勒旋转
    - 保证不同 t_offset 下是同一结构的旋转

    Args:
        n_phi: 角度方向分辨率（对应 0-2π）
        n_r: 径向方向分辨率（对应 r_inner 到 r_outer）
        seed: 随机种子
        r_inner: 内半径
        r_outer: 外半径
        enable_rt: 是否启用 Rayleigh-Taylor 不稳定性
        t_offset: 旋转偏移量（用于动画）
        color_temp: 色温（单位：K），控制整体颜色。默认 None 使用 DISK_COLOR_TEMPERATURE

    Returns:
        (n_r, n_phi, 4) float32 纹理，RGBA
    """
    generation_scale = _validate_disk_generation_scale(generation_scale)

    if state is not None:
        if state.n_phi != n_phi or state.n_r != n_r:
            raise ValueError(
                f"State size mismatch: expected {state.n_r}x{state.n_phi}, got {n_r}x{n_phi}"
            )
        if state.generation_scale != generation_scale:
            raise ValueError(
                f"State generation_scale mismatch: expected {state.generation_scale}, got {generation_scale}"
            )
        return _generate_disk_texture_rotating_from_state(state, t_offset=t_offset, color_temp=color_temp)

    # 使用全局色温参数或传入的色温
    if color_temp is None:
        color_temp = DISK_COLOR_TEMPERATURE

    rng = np.random.default_rng(seed)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_norm = np.linspace(0, 1, n_r)
    phi_grid_base, r_norm_grid = np.meshgrid(phi, r_norm)

    r_vals = r_inner + (r_outer - r_inner) * r_norm_grid
    disk_area = (r_outer ** 2 - r_inner ** 2) / 10.0

    # 计算开普勒角速度
    omega_grid = np.sqrt(0.5 / (r_vals ** 3 + 1e-6))

    # 应用旋转后的 phi_grid（开普勒旋转：逆时针，角度增加）
    phi_grid = phi_grid_base + t_offset * omega_grid

    # ----- 温度基底 -----
    radial_decay = np.clip(1.0 - r_norm_grid, 0, 1) ** 1.3
    temp_coarse = _fbm_noise((n_r, n_phi), rng, octaves=4, persistence=0.6, base_scale=8, wrap_u=True)
    temp_fine = _fbm_noise((n_r, n_phi), rng, octaves=5, persistence=0.45, base_scale=3, wrap_u=True)

    # 对温度基底应用开普勒旋转
    if t_offset != 0.0:
        for ri in range(n_r):
            rotation_pixels = int(t_offset * omega_grid[ri, 0] / (2 * np.pi) * n_phi)
            temp_coarse[ri, :] = np.roll(temp_coarse[ri, :], -rotation_pixels)
            temp_fine[ri, :] = np.roll(temp_fine[ri, :], -rotation_pixels)

    temp_noise = 0.6 * temp_coarse + 0.4 * temp_fine
    temp_base = np.clip(radial_decay * (0.85 + 0.15 * temp_noise), 0, 1)
    temp_base *= 0.25

    temp_struct = np.zeros((n_r, n_phi), dtype=np.float32)

    # ----- 密度场 -----
    # 1) 螺旋臂
    spiral, spiral_temp = _generate_spiral_arms(
        rng, n_r, n_phi, phi_grid, r_norm_grid, t_offset, omega_grid, generation_scale=generation_scale
    )
    temp_struct += spiral_temp

    # 2) 云雾
    turbulence, kep_shift_pixels, turb_temp = _generate_turbulence(
        rng, n_r, n_phi, r_norm_grid, t_offset, omega_grid, generation_scale=generation_scale
    )
    temp_struct += turb_temp

    # 3) Filaments
    arcs, arcs_temp = _generate_filaments(
        rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, t_offset, omega_grid, generation_scale=generation_scale
    )
    temp_struct += arcs_temp

    # 4) RT 不稳定性
    rt_spikes, rt_temp = _generate_rt_spikes(
        rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, enable_rt, t_offset, omega_grid, generation_scale=generation_scale
    )
    temp_struct += rt_temp

    # 5) 温度热点
    hotspot, hotspot_temp = _generate_hotspots(rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, t_offset, omega_grid)
    temp_struct += hotspot_temp

    # 6) 方位热点
    az_hotspot = _generate_azimuthal_hotspot(
        rng, n_r, n_phi, phi_grid, r_norm_grid, t_offset, omega_grid, generation_scale=generation_scale
    )

    # 组合密度
    rt_weight = 0.20 if enable_rt else 0.0
    density = 0.15 + 0.10 * spiral + 0.15 * turbulence + 0.20 * hotspot + 0.30 * arcs + rt_weight * rt_spikes

    # 湍流扰动
    density, temp_struct = _apply_disturbance(
        rng, n_r, n_phi, density, temp_struct, kep_shift_pixels, r_norm_grid, t_offset, omega_grid,
        generation_scale=generation_scale,
    )

    # 边缘软化
    edge = compute_edge_alpha(n_r)
    density *= edge[:, None]

    # 归一化
    density = np.clip(density / (np.percentile(density, 98) + 1e-6), 0, 1)

    # 合成温度场
    if np.any(temp_struct > 0):
        struct_scale = np.percentile(temp_struct[temp_struct > 0], 95)
        temp_struct_scaled = temp_struct / (struct_scale + 1e-6)
    else:
        temp_struct_scaled = temp_struct
    temp_struct_scaled = np.clip(temp_struct_scaled * 0.8, 0, 1.2)

    struct_max_per_r = np.max(temp_struct_scaled, axis=1)
    struct_p70_per_r = np.quantile(temp_struct_scaled, 0.7, axis=1)
    struct_ceiling = np.maximum(struct_p70_per_r, 0.05)
    temp_base = np.minimum(temp_base, struct_ceiling[:, None])
    temp_base = np.minimum(temp_base, struct_max_per_r[:, None])

    temperature_field = np.clip(np.maximum(temp_base, temp_struct_scaled), 0, 1)

    # 颜色（温度 -> 黑体辐射 RGB）
    # 色温控制温度映射范围：
    # - 2700K: 整体温度降低，更多区域处于红橙温度 (1500K-6000K)
    # - 4500K: 中等温度范围 (2000K-9000K)
    # - 6500K: 整体温度升高，更多区域处于白色温度 (3000K-12000K)
    # 使用线性插值：以 4500K 为基准，色温变化时调整 T_min 和 T_max
    t_factor = (color_temp - 4500) / (6500 - 2700)  # -0.47 ~ 0.47
    T_min = 2000 + t_factor * 1000  # 1500K ~ 2500K
    T_max = 9000 + t_factor * 3000  # 6000K ~ 12000K

    temp_aniso = np.clip(temperature_field * (0.9 + 0.25 * az_hotspot), 0, 1)
    T_K = T_min + temp_aniso * (T_max - T_min)
    bb_color = _blackbody_rgb(T_K)
    bb_color[:, :, 2] = np.minimum(bb_color[:, :, 2], bb_color[:, :, 0])

    luminosity = np.clip(np.sqrt(temp_aniso), 0, 1)

    tex = np.zeros((n_r, n_phi, 4), dtype=np.float32)
    tex[:, :, 0] = np.clip(bb_color[:, :, 0] * luminosity, 0, 1)
    tex[:, :, 1] = np.clip(bb_color[:, :, 1] * luminosity, 0, 1)
    tex[:, :, 2] = np.clip(bb_color[:, :, 2] * luminosity, 0, 1)
    tex[:, :, 3] = np.clip(density, 0, 1)

    return tex
