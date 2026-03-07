#!/usr/bin/env python3
"""
可视化所有组件的旋转对比
每个 t_offset 一张图，从上到下显示所有组件
使用 Pillow 绘制

关键修复：预先生成所有结构参数，保证不同 t_offset 下是同一结构的旋转
"""

import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# 导入 render.py 的函数
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render import (
    _tileable_noise,
    _fbm_noise,
)

# 参数
N_R = 64
N_PHI = 256
SEED = 42
T_OFFSETS = [0.0, 0.25, 0.5, 0.75]  # 4 个时间偏移
R_INNER = 2.0
R_OUTER = 15.0

# 创建输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "component_rows")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 删除旧的图像文件
import glob
for f in glob.glob(os.path.join(OUTPUT_DIR, "*.png")):
    os.remove(f)
    print(f"已删除旧文件：{os.path.basename(f)}")

# 组件列表
COMPONENTS = [
    ("spiral_arms", "螺旋臂"),
    ("turbulence", "湍流"),
    ("filaments", "细丝"),
    ("hotspots", "热点"),
    ("rt_spikes", "RT 不稳定性"),
    ("azimuthal_hotspot", "方位热点"),
    ("temp_base_noise", "温度基底噪声"),
    # ("disturbance_density", "扰动密度"),  # 暂时禁用 - 值太均匀，可视化效果不佳
]

print("=" * 60)
print("生成所有组件的旋转对比图")
print("=" * 60)
print(f"参数：n_r={N_R}, n_phi={N_PHI}, seed={SEED}")
print(f"时间偏移：{T_OFFSETS}")
print(f"输出目录：{OUTPUT_DIR}")
print("=" * 60)

# 创建网格
phi = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
r_norm = np.linspace(0, 1, N_R)
phi_grid_base, r_norm_grid = np.meshgrid(phi, r_norm)
r_vals = R_INNER + (R_OUTER - R_INNER) * r_norm_grid
omega_grid = np.sqrt(0.5 / (r_vals + 0.01))

# 图像参数
LABEL_WIDTH = 120
COMPONENT_HEIGHT = 100
IMG_WIDTH = 800

# 尝试加载中文字体
try:
    font = ImageFont.truetype("simhei.ttf", 16)
    title_font = ImageFont.truetype("simhei.ttf", 20)
except:
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()


def generate_spiral_arms_fixed(params, phi_grid, r_norm_grid, t_offset, omega_grid):
    """生成螺旋臂 - 使用预先生成的参数"""
    spiral = np.zeros((N_R, N_PHI), dtype=np.float32)
    temp_contribution = np.zeros((N_R, N_PHI), dtype=np.float32)

    for arm in params['arms']:
        for j in range(arm['sub_arm_count']):
            sr = arm['sub_r_starts'][j]
            sr_len = arm['sub_arm_lengths'][j]
            sr_width = arm['sub_widths'][j]
            sr_int = arm['sub_intensities'][j]
            sr_end = sr + sr_len

            # 螺旋臂角度公式（使用旋转后的 phi_grid）
            arm_angle = phi_grid - arm['base_angle'] + r_norm_grid * arm['rotations'] * 2 * np.pi

            # 宽度调制 - 使用固定的噪声（与 t_offset 无关）
            arm_noise = params['arm_noise']
            width_mod = 0.2 + 1.5 * arm_noise
            width_mod = np.clip(width_mod, 0.15, 3.0)

            arm_kappa = 1.0 / (sr_width ** 2) * 1.5
            arm_val = np.exp(arm_kappa * (np.cos(arm_angle) - 1) * width_mod)

            # 径向 mask
            mask = (r_norm_grid >= sr) & (r_norm_grid <= sr_end)
            arm_val = np.where(mask, arm_val, 0)

            # 强度调制
            intensity_mod = 0.1 + 0.9 * (arm_noise ** 0.15)

            # 边缘软化
            fade_edge = 0.02
            fade_in = np.clip((r_norm_grid - sr) / fade_edge, 0, 1)
            fade_out = np.clip((sr_end - r_norm_grid) / fade_edge, 0, 1)
            arm_val *= fade_in * fade_out * sr_int * intensity_mod

            spiral += arm_val
            temp_contribution += arm_val * arm['arm_delta_T']

    spiral = np.clip(spiral / (np.max(spiral) + 1e-6), 0, 1)
    return spiral


def generate_turbulence_fixed(params, r_norm_grid, t_offset, omega_grid):
    """生成湍流 - 使用预先生成的噪声"""
    shear_strength = params['shear_strength']
    kep_shear = shear_strength * (1.0 / (r_norm_grid + 0.3) ** 1.5 - 0.8)
    kep_shear = np.clip(kep_shear, 0, shear_strength * 8)
    kep_shift_pixels = (kep_shear / (2 * np.pi) * N_PHI).astype(int)
    max_shift = N_PHI // 4
    kep_shift_pixels = np.clip(kep_shift_pixels, -max_shift, max_shift)

    # 使用预先生成的噪声
    turbulence_coarse = params['noise_coarse']
    turbulence_mid = params['noise_mid']
    turbulence_fine = params['noise_fine']
    turbulence_extra = params['noise_extra']
    turbulence_ultra = params['noise_ultra']
    pixel_noise = params['pixel_noise']

    # 如果 t_offset != 0，应用旋转
    if t_offset != 0.0:
        for ri in range(N_R):
            rot = int(t_offset * omega_grid[ri, 0] / (2 * np.pi) * N_PHI)
            turbulence_coarse[ri, :] = np.roll(turbulence_coarse[ri, :], rot)
            turbulence_mid[ri, :] = np.roll(turbulence_mid[ri, :], rot)
            turbulence_fine[ri, :] = np.roll(turbulence_fine[ri, :], rot)
            turbulence_extra[ri, :] = np.roll(turbulence_extra[ri, :], rot)
            turbulence_ultra[ri, :] = np.roll(turbulence_ultra[ri, :], rot)
            pixel_noise[ri, :] = np.roll(pixel_noise[ri, :], rot)

    turbulence = (0.03 * turbulence_coarse + 0.05 * turbulence_mid
                  + 0.08 * turbulence_fine + 0.05 * turbulence_extra
                  + 0.03 * turbulence_ultra + 0.02 * np.clip(pixel_noise, 0, 1))

    return turbulence, kep_shift_pixels


def generate_filaments_fixed(params, phi_grid, r_norm_grid, t_offset, omega_grid):
    """生成细丝 - 使用预先生成的参数"""
    arcs = np.zeros((N_R, N_PHI), dtype=np.float32)
    temp_contribution = np.zeros((N_R, N_PHI), dtype=np.float32)

    for i in range(params['arc_count']):
        base_phi = params['arc_phi_starts'][i]
        base_r = params['arc_rs'][i]
        base_width = params['arc_r_widths'][i]
        total_length = params['arc_lengths'][i]
        intensity = params['arc_intensities'][i]
        delta_T = params['arc_delta_Ts'][i]

        sub_count = params['sub_filament_counts'][i]
        sub_lengths = params['sub_lengths_list'][i]
        sub_starts = params['sub_starts_list'][i]
        sub_widths = params['sub_widths_list'][i]
        sub_intensities = params['sub_intensities_list'][i]

        for j in range(sub_count):
            sub_phi = sub_starts[j]
            sub_len = sub_lengths[j]
            sub_w = sub_widths[j]
            sub_int = sub_intensities[j]

            phi_range = sub_len / (base_r + 0.01)
            phi_half_width = np.maximum(phi_range * 0.7, 0.2)
            kappa = 1.5 / (phi_half_width ** 2)

            sub_val = np.exp(kappa * (np.cos(phi_grid - sub_phi) - 1))

            r_diff = r_norm_grid - base_r
            r_prof = np.exp(-0.5 * (r_diff / sub_w) ** 2)

            arcs += sub_val * r_prof * sub_int
            temp_contribution += sub_val * r_prof * sub_int * delta_T * 0.7

    arcs = np.clip(arcs, 0, 1)
    temp_contribution = np.clip(temp_contribution, 0, arcs * 0.5)
    return arcs, temp_contribution


def generate_hotspots_fixed(params, phi_grid, r_norm_grid, t_offset, omega_grid):
    """生成热点 - 使用预先生成的参数"""
    hotspot = np.zeros((N_R, N_PHI), dtype=np.float32)

    for i in range(params['hotspot_count']):
        h_phi = params['h_phis'][i]
        h_r = params['h_rs'][i]
        h_phi_width = params['h_phi_widths'][i]
        h_r_width = params['h_r_widths'][i]
        h_int = params['h_intensities'][i]

        kappa = 1.0 / (h_phi_width ** 2) * 1.5
        h_batch = np.exp(kappa * (np.cos(phi_grid - h_phi) - 1.0))
        r_diff = r_norm_grid - h_r
        h_batch *= np.exp(-0.5 * (r_diff / h_r_width) ** 2)
        h_batch *= h_int

        hotspot += h_batch

    hotspot = np.clip(hotspot, 0, 1)
    return hotspot


def generate_rt_spikes_fixed(params, phi_grid, r_norm_grid, t_offset, omega_grid, enable_rt=True):
    """生成 RT 不稳定性 - 使用预先生成的参数"""
    rt_spikes = np.zeros((N_R, N_PHI), dtype=np.float32)
    temp_contribution = np.zeros((N_R, N_PHI), dtype=np.float32)

    if not enable_rt:
        return rt_spikes, temp_contribution

    for i in range(params['rt_count']):
        rt_phi = params['rt_phis'][i]
        rt_r_base = params['rt_r_bases'][i]
        rt_phi_width = params['rt_phi_widths'][i]
        rt_r_length = params['rt_r_lengths'][i]
        rt_int = params['rt_intensities'][i]
        rt_delta_T = params['rt_delta_Ts'][i]

        rt_phi_kappa = 1.0 / (rt_phi_width ** 2) * 1.5
        rt_val = np.exp(rt_phi_kappa * (np.cos(phi_grid - rt_phi) - 1))

        rt_r_diff = r_norm_grid - rt_r_base
        r_fade_out = np.clip(rt_r_length * 2 - rt_r_diff, 0, 1)
        r_fade_in = np.clip((r_norm_grid - rt_r_base) / (rt_r_length * 0.3), 0, 1)
        rt_r_profile = np.exp(-0.5 * (rt_r_diff / (rt_r_length * 0.4)) ** 2) * r_fade_out * r_fade_in

        rt_val *= rt_r_profile * rt_int
        rt_spikes += rt_val
        temp_contribution += rt_val * rt_delta_T

    rt_spikes = np.clip(rt_spikes, 0, 1)
    return rt_spikes, temp_contribution


def generate_azimuthal_hotspot_fixed(params, phi_grid, r_norm_grid, t_offset, omega_grid):
    """生成方位热点 - 使用预先生成的参数"""
    az_wave = 0.5 + 0.5 * np.sin((phi_grid + params['shear']) * params['az_freq'])
    az_noise = params['az_noise'].copy()

    if t_offset != 0.0:
        for ri in range(N_R):
            rot = int(t_offset * omega_grid[ri, 0] / (2 * np.pi) * N_PHI)
            az_noise[ri, :] = np.roll(az_noise[ri, :], rot)

    az_hotspot = np.clip(0.6 * az_wave + 0.4 * az_noise, 0, 1) ** 1.2
    return az_hotspot


def generate_temp_base_noise_fixed(params, t_offset, omega_grid):
    """生成温度基底噪声"""
    temp_coarse = params['temp_coarse'].copy()
    temp_fine = params['temp_fine'].copy()

    if t_offset != 0.0:
        for ri in range(N_R):
            rot = int(t_offset * omega_grid[ri, 0] / (2 * np.pi) * N_PHI)
            temp_coarse[ri, :] = np.roll(temp_coarse[ri, :], rot)
            temp_fine[ri, :] = np.roll(temp_fine[ri, :], rot)

    return 0.6 * temp_coarse + 0.4 * temp_fine


def generate_disturbance_density_fixed(params, r_norm_grid, t_offset, omega_grid):
    """生成扰动密度"""
    fake_density = np.ones((N_R, N_PHI)) * 0.5
    fake_temp = np.ones((N_R, N_PHI)) * 0.3

    # 使用预先生成的噪声
    disturb_coarse = params['noise_coarse'].copy()
    disturb_mid = params['noise_mid'].copy()
    disturb_fine = params['noise_fine'].copy()
    disturb_extra = params['noise_extra'].copy()
    disturb_pixel = params['pixel_noise'].copy()

    if t_offset != 0.0:
        for ri in range(N_R):
            rot = int(t_offset * omega_grid[ri, 0] / (2 * np.pi) * N_PHI)
            disturb_coarse[ri, :] = np.roll(disturb_coarse[ri, :], rot)
            disturb_mid[ri, :] = np.roll(disturb_mid[ri, :], rot)
            disturb_fine[ri, :] = np.roll(disturb_fine[ri, :], rot)
            disturb_extra[ri, :] = np.roll(disturb_extra[ri, :], rot)
            disturb_pixel[ri, :] = np.roll(disturb_pixel[ri, :], rot)

    shear_strength = 4.0
    kep_shear = shear_strength * (1.0 / (r_norm_grid + 0.3) ** 1.5 - 0.8)
    kep_shear = np.clip(kep_shear, 0, shear_strength * 8)
    kep_shift_pixels = (kep_shear / (2 * np.pi) * N_PHI).astype(int)
    max_shift = N_PHI // 4
    kep_shift_pixels = np.clip(kep_shift_pixels, -max_shift, max_shift)

    disturb_mod = (0.01 * disturb_coarse + 0.02 * disturb_mid + 0.04 * disturb_fine
                   + 0.03 * disturb_extra + 0.02 * disturb_pixel)
    disturb_mod = np.clip(disturb_mod * 0.5 + 0.5, 0.5, 1.0)
    radial_preserve = 0.6 + 0.4 * r_norm_grid
    disturb_mod = np.clip(disturb_mod * radial_preserve, 0.5, 1.0)
    density = fake_density * disturb_mod

    return density


# ============================================================
# 预先生成所有组件的随机参数（保证不同 t_offset 下结构相同）
# ============================================================
print("\n预先生成所有组件的随机参数...")
rng = np.random.default_rng(SEED)

# --- 螺旋臂参数 ---
print("  生成螺旋臂参数...")
n_arms = rng.integers(2, 5)
n_from_center = rng.integers(2, 4)
spiral_params = {'n_arms': n_arms, 'arms': []}

# 预先生成 arm_noise（用于所有 sub-arm）
arm_noise = _tileable_noise((N_R, N_PHI), rng, freq_u=3, freq_v=2)

for arm_idx in range(n_arms):
    if arm_idx < n_from_center:
        r_start = 0.0
        base_angle = arm_idx * 2 * np.pi / n_from_center
    else:
        r_start = rng.uniform(0.05, 0.5)
        base_angle = rng.uniform(0, 2 * np.pi)

    rotations = rng.uniform(2.5, 5.0)
    base_width = rng.uniform(0.2, 0.4)
    arm_delta_T = rng.uniform(0.1, 0.3)
    r_length = rotations / 6.0 * (1.0 - r_start)
    r_length = min(r_length, 1.0 - r_start)

    sub_arm_count = rng.integers(4, 9)
    sub_arm_fill = rng.uniform(0.4, 0.6)
    sub_arm_lengths = rng.uniform(0.08, 0.20, sub_arm_count)
    sub_arm_lengths = sub_arm_lengths / sub_arm_lengths.sum() * r_length * sub_arm_fill

    sub_r_starts = np.zeros(sub_arm_count)
    for j in range(1, sub_arm_count):
        gap = rng.uniform(0.08, 0.15)
        sub_r_starts[j] = sub_r_starts[j-1] + sub_arm_lengths[j-1] + gap
    sub_r_starts += r_start

    sub_widths = base_width * rng.uniform(0.3, 2.5, sub_arm_count)
    sub_widths = np.clip(sub_widths, 0.06, 1.2)
    sub_intensities = rng.uniform(0.1, 0.7, sub_arm_count)

    spiral_params['arms'].append({
        'r_start': r_start, 'base_angle': base_angle, 'rotations': rotations,
        'base_width': base_width, 'arm_delta_T': arm_delta_T,
        'sub_arm_count': sub_arm_count, 'sub_arm_lengths': sub_arm_lengths,
        'sub_r_starts': sub_r_starts, 'sub_widths': sub_widths,
        'sub_intensities': sub_intensities
    })

spiral_params['arm_noise'] = arm_noise

# --- 湍流参数 ---
print("  生成湍流参数...")
shear_strength = rng.uniform(3.0, 6.0)
turbulence_params = {
    'shear_strength': shear_strength,
    'noise_coarse': _tileable_noise((N_R, N_PHI), rng, freq_u=8, freq_v=4),
    'noise_mid': _tileable_noise((N_R, N_PHI), rng, freq_u=24, freq_v=12),
    'noise_fine': _tileable_noise((N_R, N_PHI), rng, freq_u=80, freq_v=40),
    'noise_extra': _tileable_noise((N_R, N_PHI), rng, freq_u=200, freq_v=100),
    'noise_ultra': _tileable_noise((N_R, N_PHI), rng, freq_u=400, freq_v=200),
    'pixel_noise': (_tileable_noise((N_R, N_PHI), rng, freq_u=600, freq_v=300) - 0.5) * 2
}

# --- 细丝参数 ---
print("  生成细丝参数...")
arc_count = int(rng.uniform(80, 150))
filament_params = {
    'arc_count': arc_count,
    'sub_filament_counts': rng.integers(2, 5, arc_count),
    'arc_phi_starts': rng.uniform(0, 2 * np.pi, arc_count),
    'arc_rs': 0.05 + rng.uniform(0.05, 0.95, arc_count) ** 0.6 * 0.9,
    'arc_r_widths': rng.uniform(0.002, 0.008, arc_count),
    'arc_lengths': rng.uniform(0.5, 1.2, arc_count),
    'arc_intensities': rng.uniform(0.7, 1.0, arc_count),
    'arc_delta_Ts': 0.3 + 0.6 * rng.power(0.3, arc_count),
    'sub_lengths_list': [],
    'sub_starts_list': [],
    'sub_widths_list': [],
    'sub_intensities_list': []
}

for i in range(arc_count):
    sub_count = filament_params['sub_filament_counts'][i]
    total_length = filament_params['arc_lengths'][i]
    sub_fill = rng.uniform(0.35, 0.55)
    sub_lengths = rng.uniform(0.08, 0.20, sub_count)
    sub_lengths = sub_lengths / sub_lengths.sum() * total_length * sub_fill

    sub_starts = np.zeros(sub_count)
    sub_starts[0] = filament_params['arc_phi_starts'][i]
    for j in range(1, sub_count):
        gap = rng.uniform(0.08, 0.20)
        sub_starts[j] = sub_starts[j-1] + sub_lengths[j-1] + gap

    base_width = filament_params['arc_r_widths'][i]
    sub_widths = base_width * rng.uniform(0.3, 3.0, sub_count)
    sub_widths = np.clip(sub_widths, 0.001, 0.025)
    sub_intensities = filament_params['arc_intensities'][i] * rng.uniform(0.15, 1.0, sub_count)

    filament_params['sub_lengths_list'].append(sub_lengths)
    filament_params['sub_starts_list'].append(sub_starts)
    filament_params['sub_widths_list'].append(sub_widths)
    filament_params['sub_intensities_list'].append(sub_intensities)

# --- 热点参数 ---
print("  生成热点参数...")
hotspot_count = int(rng.uniform(20, 40))
hotspot_params = {
    'hotspot_count': hotspot_count,
    'h_phis': rng.uniform(0, 2 * np.pi, hotspot_count),
    'h_rs': 0.1 + rng.uniform(0, 1, hotspot_count) ** 0.6 * 0.85,
    'h_phi_widths': rng.uniform(0.08, 0.20, hotspot_count),
    'h_r_widths': 0.02 + rng.uniform(0, 0.03, hotspot_count),
    'h_intensities': np.zeros(hotspot_count)  # 稍后计算
}
# 重新计算 h_intensities（依赖于 h_rs）
hotspot_params['h_intensities'] = 0.3 + (1 - hotspot_params['h_rs']) * 0.6 + rng.uniform(0, 0.1, hotspot_count)

# --- RT 不稳定性参数 ---
print("  生成 RT 不稳定性参数...")
rt_count = int(rng.uniform(15, 30) * 0.8)  # 使用固定的 disk_area 近似值
rt_params = {
    'rt_count': rt_count,
    'rt_phis': rng.uniform(0, 2 * np.pi, rt_count),
    'rt_r_bases': np.power(rng.uniform(0.01, 0.15, rt_count), 1.5),
    'rt_phi_widths': rng.uniform(0.08, 0.20, rt_count),
    'rt_r_lengths': rng.uniform(0.08, 0.20, rt_count),
    'rt_intensities': rng.uniform(0.8, 1.0, rt_count),
    'rt_delta_Ts': rng.uniform(0.5, 1.2, rt_count)
}

# --- 方位热点参数 ---
print("  生成方位热点参数...")
az_freq = rng.integers(2, 5)
azimuthal_params = {
    'az_freq': az_freq,
    'shear': r_norm_grid ** 1.2 * rng.uniform(2.0, 4.0),
    'az_noise': _fbm_noise((N_R, N_PHI), rng, octaves=3, persistence=0.5, base_scale=3, wrap_u=True)
}

# --- 温度基底噪声参数 ---
print("  生成温度基底噪声参数...")
temp_base_params = {
    'temp_coarse': _fbm_noise((N_R, N_PHI), rng, octaves=4, persistence=0.6, base_scale=8, wrap_u=True),
    'temp_fine': _fbm_noise((N_R, N_PHI), rng, octaves=5, persistence=0.45, base_scale=3, wrap_u=True)
}

# --- 扰动密度参数 ---
print("  生成扰动密度参数...")
disturbance_params = {
    'noise_coarse': _tileable_noise((N_R, N_PHI), rng, freq_u=8, freq_v=4),
    'noise_mid': _tileable_noise((N_R, N_PHI), rng, freq_u=32, freq_v=16),
    'noise_fine': _tileable_noise((N_R, N_PHI), rng, freq_u=100, freq_v=50),
    'noise_extra': _tileable_noise((N_R, N_PHI), rng, freq_u=250, freq_v=125),
    'pixel_noise': (_tileable_noise((N_R, N_PHI), rng, freq_u=400, freq_v=200) - 0.5) * 2
}

# 存储所有参数的字典
ALL_PARAMS = {
    'spiral_arms': spiral_params,
    'turbulence': turbulence_params,
    'filaments': filament_params,
    'hotspots': hotspot_params,
    'azimuthal_hotspot': azimuthal_params,
    'temp_base_noise': temp_base_params,
    'disturbance_density': disturbance_params
}

print("参数生成完成！\n")


def normalize_for_display(data):
    """归一化数据用于显示"""
    if data.max() == data.min():
        return np.zeros_like(data)
    return (data - data.min()) / (data.max() - data.min())


def data_to_image(data, width=600, height=80):
    """将数据数组转换为 PIL 图像"""
    from PIL import Image
    n_r, n_phi = data.shape

    # 缩放到目标高度
    scale = height / n_r
    img_width = int(n_phi * scale)

    # 双线性插值缩放
    scaled = np.zeros((height, img_width), dtype=np.float32)
    for y in range(height):
        for x in range(img_width):
            src_y = y / height * n_r
            src_x = x / img_width * n_phi

            y0, x0 = int(src_y), int(src_x)
            y1, x1 = min(y0 + 1, n_r - 1), min(x0 + 1, n_phi - 1)

            dy, dx = src_y - y0, src_x - x0

            v00 = data[y0, x0]
            v01 = data[y0, x1]
            v10 = data[y1, x0]
            v11 = data[y1, x1]

            v0 = v00 * (1 - dx) + v01 * dx
            v1 = v10 * (1 - dx) + v11 * dx
            v = v0 * (1 - dy) + v1 * dy

            scaled[y, x] = v

    # 转换为灰度图像
    gray = (scaled * 255).astype(np.uint8)
    img = Image.fromarray(gray, mode='L')

    # 调整到目标宽度
    img = img.resize((width, height), Image.Resampling.BILINEAR)

    return img


# ============================================================
# 为每个 t_offset 生成图像
# ============================================================
for t_idx, t_offset in enumerate(T_OFFSETS):
    print(f"生成 t_offset = {t_offset:.2f} 的图像...")

    # 计算旋转后的 phi_grid
    phi_grid = phi_grid_base - t_offset * omega_grid

    # 生成所有组件的数据
    component_data = {}

    # 螺旋臂
    component_data['spiral_arms'] = generate_spiral_arms_fixed(
        spiral_params, phi_grid, r_norm_grid, t_offset, omega_grid)

    # 湍流
    turb, _, = generate_turbulence_fixed(
        turbulence_params, r_norm_grid, t_offset, omega_grid)
    component_data['turbulence'] = turb

    # 细丝
    component_data['filaments'] = generate_filaments_fixed(
        filament_params, phi_grid, r_norm_grid, t_offset, omega_grid)[0]

    # 热点
    component_data['hotspots'] = generate_hotspots_fixed(
        hotspot_params, phi_grid, r_norm_grid, t_offset, omega_grid)

    # RT 不稳定性
    component_data['rt_spikes'] = generate_rt_spikes_fixed(
        rt_params, phi_grid, r_norm_grid, t_offset, omega_grid, enable_rt=True)[0]

    # 方位热点
    component_data['azimuthal_hotspot'] = generate_azimuthal_hotspot_fixed(
        azimuthal_params, phi_grid, r_norm_grid, t_offset, omega_grid)

    # 温度基底噪声
    component_data['temp_base_noise'] = generate_temp_base_noise_fixed(
        temp_base_params, t_offset, omega_grid)

    # 扰动密度
    component_data['disturbance_density'] = generate_disturbance_density_fixed(
        disturbance_params, r_norm_grid, t_offset, omega_grid)

    # 创建图像
    total_height = len(COMPONENTS) * COMPONENT_HEIGHT + 40
    img = Image.new('L', (LABEL_WIDTH + IMG_WIDTH, total_height), color=255)
    draw = ImageDraw.Draw(img)

    # 绘制标题
    title = f"t_offset = {t_offset:.2f}"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = bbox[2] - bbox[0]
    draw.text(((LABEL_WIDTH + IMG_WIDTH - title_width) / 2, 10), title, fill=0, font=title_font)

    # 绘制每个组件
    for i, (name, label) in enumerate(COMPONENTS):
        y_offset = 40 + i * COMPONENT_HEIGHT

        # 绘制左侧标签
        draw.text((10, y_offset + COMPONENT_HEIGHT / 2 - 8), label, fill=0, font=font)

        # 绘制组件图像
        data = normalize_for_display(component_data[name])
        comp_img = data_to_image(data, IMG_WIDTH, COMPONENT_HEIGHT)
        img.paste(comp_img, (LABEL_WIDTH, y_offset))

    # 保存
    output_path = os.path.join(OUTPUT_DIR, f"components_t{t_offset:.2f}.png")
    img.save(output_path)
    print(f"  已保存：{output_path}")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
