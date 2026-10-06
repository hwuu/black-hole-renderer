"""运镜路径：关键帧 → 光滑空间路径 → 恒定感知速度的节奏 → 逐帧相机状态。

设计见 `docs/plans/v2_camera_path_plan.md` §5.2–§5.4。路径分两层：

1. **空间路径** C(p)：关键帧决定相机经过哪里。p ∈ [0, K − 1] 为关键帧序号的连续化；
   7 个配置量 (ln r, φ, z, fov, roll, u, v) 先做保形三次插值，再沿 p 做高斯平滑，最后用三次样条还原为连续函数。
2. **节奏** p(t)：决定何时经过。沿路径累计感知进度 S（画面变化量），感知速度 dS/dt 全程恒定，
   只在开头与结尾各缓变一次。相机位置、速度、加速度对时间连续。

黑洞（原点）是唯一主体：每帧的相机光轴由"黑洞在画面中的位置 (u, v)"反解（`camera_compose.solve_forward`）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .camera_compose import UniformSpline, gaussian_smooth, pchip, solve_forward

# 空间路径网格：每个关键帧间隔的采样点数
GRID_PER_KEYFRAME = 400
# 节奏曲线的时间网格步长（视频秒）
TIME_STEP = 1.0 / 240.0
# 感知进度按固定宽高比计算转头角，使节奏与输出分辨率无关
PROGRESS_ASPECT = 16.0 / 9.0
# 路径检查：采样间隔（视频秒）、光轴与竖直方向的最小夹角（度）、离黑洞的最小距离（r_s）
CHECK_STEP = 0.1
CHECK_MIN_TILT_DEG = 1.0
CHECK_MIN_DISTANCE = 3.0
# 最短路径时长（视频秒）：节奏曲线的时间网格至少需要 4 个节点（三次样条），即 3 个时间步
MIN_DURATION = 3 * TIME_STEP
# 路径报告：速度的中心差分半窗口（视频秒）、扫描盘面穿越与最大速度的采样间隔（视频秒）
REPORT_SPEED_HALF_WINDOW = 0.05
REPORT_SCAN_STEP = 0.1
# 黑洞目标画面位置 (u, v) 的取值范围：[0, 1] 为画面内；越出时黑洞在画面外（如长焦特写只拍盘的一侧），
# 最多偏出一个画面宽 / 高，避免光轴远离黑洞时构图求解失去意义
SUBJECT_UV_MIN = -1.0
SUBJECT_UV_MAX = 2.0


@dataclass(frozen=True)
class Keyframe:
    """路径控制点：相机经过的一个状态（不带时刻，经过时刻由节奏决定）。

    Attributes:
        r: 柱半径（r_s，> 0）。
        phi_deg: 方位角 φ（度），与 `--pov` 的 `atan2(y, x)` 一致；跨关键帧连续取值（不折回 ±180°）。
        z: 离盘面高度（r_s），负值为盘下方。
        fov: 竖直视野角（度，(0, 180)）。
        roll: 相机滚转角（度），正值使画面左低右高。
        subject_uv: 黑洞在画面中的目标位置 `(u, v)`，u 从左到右、v 从上到下；[0, 1] 为画面内，
            允许范围 [`SUBJECT_UV_MIN`, `SUBJECT_UV_MAX`] = [−1, 2]，越出 [0, 1] 时黑洞在画面外。
        label: 镜头标签，只用于报告与联系表。
    """

    r: float
    phi_deg: float
    z: float
    fov: float
    roll: float
    subject_uv: Tuple[float, float]
    label: str = ""


@dataclass(frozen=True)
class PathTiming:
    """空间平滑、感知进度与节奏的参数（默认值为示例路径的定稿值）。

    Attributes:
        duration: 视频时长 T（秒，> 0）。
        path_smoothing: 空间路径高斯平滑标准差 σ_p（关键帧间隔，≥ 0）；0 = 只做保形插值。
        near_weight: 近景视差权重 c_near（≥ 0）：贴近盘面时同样位移显得更快的程度。
        near_height_floor: 近景视差项的高度下限 z_min（r_s，> 0），防止 z → 0 时发散。
        progress_smoothing: 感知进度被积函数的高斯平滑标准差（关键帧间隔，≥ 0），消除 max / |·| 拐点。
        ramp_in: 起步缓变时长（秒，≥ 0）。
        ramp_out: 收尾缓变时长（秒，≥ 0），`ramp_in + ramp_out ≤ duration`。
        speed_start: 首帧感知速度与巡航感知速度之比，(0, 1]。
        speed_end: 末帧感知速度与巡航感知速度之比，(0, 1]。
    """

    duration: float
    path_smoothing: float = 0.5
    near_weight: float = 0.03
    near_height_floor: float = 0.3
    progress_smoothing: float = 0.3
    ramp_in: float = 6.0
    ramp_out: float = 8.0
    speed_start: float = 0.55
    speed_end: float = 0.55


@dataclass(frozen=True)
class CameraState:
    """某一视频时刻的相机状态。

    Attributes:
        pos: 相机位置（r_s），形状 `(3,)`。
        forward: 单位光轴，形状 `(3,)`。
        fov: 竖直视野角（度）。
        roll: 滚转角（度）。
        subject_uv: 黑洞的目标画面位置 `(u, v)`。
    """

    pos: np.ndarray
    forward: np.ndarray
    fov: float
    roll: float
    subject_uv: Tuple[float, float]


def _smootherstep(s: np.ndarray) -> np.ndarray:
    """五次缓动曲线，s 截断到 [0, 1]。

    Args:
        s: 归一化时间，任意形状。

    Returns:
        同形状数组，值域 [0, 1]，s ≤ 0 为 0、s ≥ 1 为 1。

    Formula:
        ss(s) = 6s⁵ − 15s⁴ + 10s³；两端一、二阶导数为 0。
    """
    s = np.clip(s, 0.0, 1.0)
    return s**3 * (10 - 15 * s + 6 * s * s)


def _position(cfg: np.ndarray) -> np.ndarray:
    """配置量 (ln r, φ°, z, …) → 笛卡尔位置。

    Args:
        cfg: 配置数组，形状 `(N, 7)`，列依次为 ln r、φ（度）、z、fov、roll、u、v。

    Returns:
        相机位置（r_s），形状 `(N, 3)`。

    Formula:
        x = r·cos φ，y = r·sin φ，z = z，r = exp(ln r)。
    """
    r, phi = np.exp(cfg[:, 0]), np.radians(cfg[:, 1])
    return np.stack([r * np.cos(phi), r * np.sin(phi), cfg[:, 2]], axis=1)


def local_speed(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """相机相对当地静止观者的速度（以光速为单位）。

    Args:
        pos: 相机位置（r_s），形状 `(N, 3)`，|pos| > 1（视界外）。
        vel: 坐标速度 dx/dt_phys（r_s 每 r_s/c，即以光速为单位），形状 `(N, 3)`。

    Returns:
        形状 `(N,)` 的非负数组；≥ 1 表示超光速，真实飞行的相机无法做到。

    Formula:
        A = 1 − 1/ρ，ρ = |x|；v_r = v·x̂，v_⊥ = |v − v_r·x̂|；
        v_loc = √((v_r / A)² + v_⊥² / A)（史瓦西度规下静止观者测得的局部速度，r_s = 1）。

    Physical Meaning:
        判断运镜是否对应一台真实飞行的相机：渲染采用静止观者近似，只有 v_loc ≪ 1 时画面才接近真实相机所见。

    Simplifications:
        把渲染坐标视为史瓦西坐标的笛卡尔化（与光线积分一致）。
    """
    rho = np.linalg.norm(pos, axis=1)
    a = 1.0 - 1.0 / rho
    v_r = np.sum(vel * pos, axis=1) / rho
    v_perp = np.linalg.norm(vel - v_r[:, None] * pos / rho[:, None], axis=1)
    return np.sqrt((v_r / a) ** 2 + v_perp**2 / a)


class CameraPath:
    """运镜路径：给定视频时刻返回相机状态。

    Args:
        keyframes: 至少 2 个关键帧（按经过顺序）。
        timing: 平滑、感知进度与节奏参数。

    Raises:
        ValueError: 关键帧或参数取值非法（见 `_validate`），或构图求解不收敛（见 `solve_forward`）。
    """

    def __init__(self, keyframes: List[Keyframe], timing: PathTiming):
        """校验参数，构造空间路径与节奏曲线。

        Args:
            keyframes: 至少 2 个关键帧（按经过顺序）。
            timing: 平滑、感知进度与节奏参数。
        """
        _validate(keyframes, timing)
        self.keyframes = list(keyframes)
        self.timing = timing
        self._build_space()
        self._build_rhythm()

    def _build_space(self) -> None:
        """构造空间路径：保形插值 + 沿 p 高斯平滑 + 三次样条还原。

        结果写入 `self._p`（p 网格，形状 `(N,)`）、`self._cfg`（网格上的配置量，形状 `(N, 7)`）与
        `self._cfg_of_p`（配置量关于 p 的连续函数）。

        Formula:
            q̃(p) = Σ q(p′)·G_σ(p − p′)，σ = path_smoothing（关键帧间隔），q 为 7 个配置量的保形插值。
        """
        kf = self.keyframes
        n_kf = len(kf)
        nodes = np.array([[math.log(k.r), k.phi_deg, k.z, k.fov, k.roll, k.subject_uv[0], k.subject_uv[1]]
                          for k in kf])
        idx = np.arange(n_kf, dtype=np.float64)
        self._p = np.linspace(0.0, n_kf - 1, (n_kf - 1) * GRID_PER_KEYFRAME + 1)
        raw = np.stack([pchip(idx, nodes[:, c], self._p) for c in range(7)], axis=1)
        self._cfg = gaussian_smooth(raw, self.timing.path_smoothing * GRID_PER_KEYFRAME)
        self._cfg_of_p = UniformSpline(0.0, 1.0 / GRID_PER_KEYFRAME, self._cfg)

    def _speed_weight(self, t: np.ndarray) -> np.ndarray:
        """节奏曲线的形状 w(t)：感知速度与巡航感知速度之比。

        Args:
            t: 视频时刻（秒），任意形状。

        Returns:
            同形状数组，值域 [min(v_s, v_e), 1]；中段为 1。

        Formula:
            w(t) = 1 − (1 − v_s)·(1 − ss(t / T_in)) − (1 − v_e)·(1 − ss((T − t) / T_out))
        """
        tm = self.timing
        return (1.0 - (1.0 - tm.speed_start) * (1.0 - _smootherstep(t / max(tm.ramp_in, 1e-12)))
                - (1.0 - tm.speed_end) * (1.0 - _smootherstep((tm.duration - t) / max(tm.ramp_out, 1e-12))))

    def _build_rhythm(self) -> None:
        """构造感知进度 S(p) 与节奏 p(t)。

        结果写入 `self._s`（p 网格上的 S，形状 `(N,)`）、`self._t` 与 `self._s_of_t`（时间网格上的 S(t)）、
        `self.cruise_speed`（巡航感知速度 c）与 `self._p_of_time`（p 关于 t 的连续函数）。

        Formula:
            dS = |dx|·(1/ρ + c_near / max(|z|, z_min)) + dθ + ½·|dfov| / fov，
            ρ = |x| 为到黑洞距离，dθ 为相邻光轴夹角；dS 沿 p 高斯平滑后累加得 S(p)。
            v(t) = c·w(t)，c = S_total / ∫₀ᵀ w dt；S(t) = c·∫₀ᵗ w dt；p(t) 满足 S(p(t)) = S(t)。

        Physical Meaning:
            S 衡量画面变了多少：离黑洞越近、越贴近盘面，同样的位移在画面中扫过的角度越大；
            转头与变焦也算在内。恒定的 dS/dt 使观众感到的运动速度恒定，相机在近处自然放慢。

        Simplifications:
            近景视差只按离盘面高度估计（不看实际气体密度）；转头角用固定宽高比 16:9 计算。
        """
        tm = self.timing
        cfg = self._cfg
        pos = _position(cfg)
        fwd = solve_forward(pos, cfg[:, 5:7], cfg[:, 3], PROGRESS_ASPECT, cfg[:, 4])
        dx = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        rho = np.linalg.norm(pos[:-1], axis=1)
        near = tm.near_weight / np.maximum(np.abs(pos[:-1, 2]), tm.near_height_floor)
        # 夹角用 atan2(|a × b|, a·b)：小角度时比 arccos(a·b) 数值稳定
        turn = np.arctan2(np.linalg.norm(np.cross(fwd[:-1], fwd[1:]), axis=1), np.sum(fwd[1:] * fwd[:-1], axis=1))
        zoom = 0.5 * np.abs(np.diff(cfg[:, 3])) / cfg[:-1, 3]
        ds = gaussian_smooth(dx * (1.0 / rho + near) + turn + zoom, tm.progress_smoothing * GRID_PER_KEYFRAME)
        self._s = np.concatenate([[0.0], np.cumsum(ds)])
        s_of_p = UniformSpline(0.0, 1.0 / GRID_PER_KEYFRAME, self._s)

        self._t = np.linspace(0.0, tm.duration, int(round(tm.duration / TIME_STEP)) + 1)
        w = self._speed_weight(self._t)
        w_int = np.concatenate([[0.0], np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(self._t))])
        self.cruise_speed = float(self._s[-1] / w_int[-1])
        self._s_of_t = w_int * self.cruise_speed
        # 反解 S(p) = S(t)：线性插值给初值，再用样条做两步牛顿迭代，使 p(t) 与光滑的 S(p) 一致
        # （dS/dp > 0：相机沿路径始终在动；下限只防端点处数值上的极小值）
        p = np.interp(self._s_of_t, self._s, self._p)
        for _ in range(2):
            slope = np.maximum(s_of_p.derivative(p), 1e-12)
            p = np.clip(p - (s_of_p(p) - self._s_of_t) / slope, 0.0, self._p[-1])
        self._p_of_time = UniformSpline(0.0, self._t[1] - self._t[0], p)

    @property
    def duration(self) -> float:
        """视频时长 T（秒，> 0），即路径文件的 `duration`。"""
        return self.timing.duration

    @property
    def progress_total(self) -> float:
        """整条路径的感知进度 S_total（无量纲，> 0）。"""
        return float(self._s[-1])

    def perceived_speed(self, t) -> np.ndarray:
        """感知速度 v(t) = c·w(t)（每视频秒的感知进度）。

        Args:
            t: 视频时刻（秒），标量或数组。

        Returns:
            与 t 同形状的正数组；中段等于 `cruise_speed`，首末分别为 `speed_start`、`speed_end` 倍。
        """
        return self.cruise_speed * self._speed_weight(np.asarray(t, dtype=np.float64))

    def config_at(self, t: float) -> np.ndarray:
        """时刻 t 的配置量。

        Args:
            t: 视频时刻（秒），截断到 [0, duration]。

        Returns:
            形状 `(7,)`：ln r（r 为柱半径，r_s）、φ（度）、z（r_s）、fov（度）、roll（度）、u、v。
        """
        return self._cfg_of_p(self._p_of_time(min(max(float(t), 0.0), self.duration)))

    def state_at(self, t: float, aspect: float) -> CameraState:
        """时刻 t 的相机状态。

        Args:
            t: 视频时刻（秒），截断到 [0, duration]。
            aspect: 输出画面宽高比 W / H（构图求解用）。

        Returns:
            `CameraState`；把它的 `pos`、`forward`、`fov`、`roll` 交给渲染器即可。
        """
        cfg = self.config_at(t)[None, :]
        pos = _position(cfg)
        fwd = solve_forward(pos, cfg[:, 5:7], cfg[:, 3], aspect, cfg[:, 4])
        return CameraState(pos=pos[0], forward=fwd[0], fov=float(cfg[0, 3]), roll=float(cfg[0, 4]),
                           subject_uv=(float(cfg[0, 5]), float(cfg[0, 6])))

    def keyframe_times(self) -> np.ndarray:
        """各关键帧（p = i）的经过时刻。

        Returns:
            形状 `(K,)` 的数组（视频秒），单调递增，首项为 0、末项为 duration。
        """
        s_kf = np.interp(np.arange(len(self.keyframes), dtype=np.float64), self._p, self._s)
        return np.interp(s_kf, self._s_of_t, self._t)

    def check(self, aspect: float) -> None:
        """每 `CHECK_STEP` 秒采样（含终点）检查相机状态。

        Args:
            aspect: 输出画面宽高比 W / H（光轴随宽高比略有不同）。

        Raises:
            ValueError: 光轴与竖直方向夹角小于 `CHECK_MIN_TILT_DEG`（画面水平方向无法由世界上方向确定），
                或相机离黑洞小于 `CHECK_MIN_DISTANCE` r_s（最内稳定圆轨道以内的强引力区，盘模型不覆盖），
                或构图求解不收敛。
        """
        cos_max = math.cos(math.radians(CHECK_MIN_TILT_DEG))
        for t in np.append(np.arange(0.0, self.duration, CHECK_STEP), self.duration):
            st = self.state_at(float(t), aspect)
            if abs(st.forward[2]) > cos_max:
                raise ValueError(f"相机路径 t={t:.1f}s：光轴接近竖直（与 ±z 夹角 < {CHECK_MIN_TILT_DEG}°）")
            if np.linalg.norm(st.pos) < CHECK_MIN_DISTANCE:
                raise ValueError(f"相机路径 t={t:.1f}s：离黑洞不足 {CHECK_MIN_DISTANCE} r_s")

    def _kinematics(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """时刻 t 的配置量、位置与坐标速度（中心差分，半窗口 `REPORT_SPEED_HALF_WINDOW`）。

        Args:
            t: 视频时刻（秒）。

        Returns:
            `(cfg, pos, vel)`：配置量 `(7,)`，位置 `(3,)`（r_s），速度 `(3,)`（r_s / 视频秒）；
            cfg 的第 1 列（φ）替换为 dφ/dt（度 / 视频秒）。
        """
        h = REPORT_SPEED_HALF_WINDOW
        ta, tb = max(t - h, 0.0), min(t + h, self.duration)
        ca, cb = self.config_at(ta), self.config_at(tb)
        pa, pb = _position(ca[None, :])[0], _position(cb[None, :])[0]
        cfg = 0.5 * (ca + cb)
        cfg[1] = (cb[1] - ca[1]) / (tb - ta)
        return cfg, 0.5 * (pa + pb), (pb - pa) / (tb - ta)

    def report(self, time_scale: float, r_in: float, r_out: float, disk_spin: float) -> List[str]:
        """路径报告：每个关键帧的经过时刻、位置、速度与相对气体角速度，全程最大局部速度，以及穿越盘面（z = 0）的时刻。

        Args:
            time_scale: 物理时间与视频时间之比 dt_phys / dt_video（(r_s/c) / 视频秒，见 `path_video.time_scale`）。
            r_in: 盘内缘半径（r_s）。
            r_out: 盘外缘半径（r_s），柱半径在 (r_in, r_out) 之外的位置没有气体。
            disk_spin: 盘旋转方向，+1 = φ 增大方向，−1 = 反向（`--v2_reverse_rotation`）。

        Returns:
            报告文本行列表。

        Formula:
            气体角速度（视频时间）ω_gas = disk_spin·√(0.5 / r³)·time_scale（开普勒角速度，r_s = 1）；
            相对角速度 ω_rel = dφ_cam/dt − ω_gas；dφ_cam/dt 与 disk_spin 异号为逆行、同号为顺行；
            局部速度见 `local_speed`，坐标速度先除以 time_scale 换算为光速单位。
        """
        lines = [f"[运镜] 时长 {self.duration:.1f} s，巡航感知速度 {self.cruise_speed:.4f}/s"]
        for k, t in zip(self.keyframes, self.keyframe_times()):
            cfg, _pos, vel = self._kinematics(float(t))
            r_c, dphi = math.exp(cfg[0]), cfg[1]
            speed = float(np.linalg.norm(vel))
            sense = "逆行" if dphi * disk_spin < 0 else "顺行"
            if r_in < r_c < r_out:
                omega_gas = disk_spin * math.degrees(math.sqrt(0.5 / r_c**3)) * time_scale
                gas = f"相对气体 {dphi - omega_gas:+5.2f}°/s"
            else:
                gas = "此处无气体"
            lines.append(f"  t={t:5.1f}s  {k.label:<8} r={r_c:5.1f} z={cfg[2]:+6.2f}  速度 {speed:5.2f} r_s/s  "
                         f"方位 {dphi:+5.2f}°/s（{sense}），{gas}")
        ts = np.arange(0.0, self.duration + 1e-9, REPORT_SCAN_STEP)
        kin = [self._kinematics(float(t)) for t in ts]
        pos = np.array([k[1] for k in kin])
        v_loc = local_speed(pos, np.array([k[2] for k in kin]) / time_scale)
        i_max = int(np.argmax(v_loc))
        lines.append(f"  最大局部速度 {v_loc[i_max]:.2f} c（t={ts[i_max]:.1f}s；渲染按静止观者近似，不含相机运动带来的光行差与频移）")
        z = np.array([k[0][2] for k in kin])
        for i in np.nonzero(np.sign(z[1:]) != np.sign(z[:-1]))[0]:
            r_c = math.exp(kin[i + 1][0][0])
            where = "穿过盘内" if r_in < r_c < r_out else "盘外，无气体"
            lines.append(f"  穿越盘面 z = 0：t={ts[i + 1]:.1f}s  r={r_c:.1f}（{where}）")
        return lines


def _validate(keyframes: List[Keyframe], timing: PathTiming) -> None:
    """校验关键帧与节奏参数（所有数值必须有限）。

    Args:
        keyframes: 关键帧列表。
        timing: 节奏参数。

    Raises:
        ValueError: 信息中给出字段名。关键帧少于 2 个；任一数值不是有限数；r ≤ 0；fov 不在 (0, 180)；
            subject_uv 不在 [−1, 2]；duration < `MIN_DURATION`；path_smoothing、progress_smoothing、near_weight、ramp_in、
            ramp_out < 0；near_height_floor ≤ 0；ramp_in + ramp_out 超过 duration；speed_start / speed_end 不在 (0, 1]。
    """
    if len(keyframes) < 2:
        raise ValueError("相机路径至少需要 2 个关键帧")
    for i, k in enumerate(keyframes):
        for name, v in (("r", k.r), ("phi_deg", k.phi_deg), ("z", k.z), ("fov", k.fov), ("roll", k.roll),
                        ("subject_uv[0]", k.subject_uv[0]), ("subject_uv[1]", k.subject_uv[1])):
            if not math.isfinite(v):
                raise ValueError(f"关键帧 {i}：{name} 必须是有限数，得到 {v}")
        if not k.r > 0:
            raise ValueError(f"关键帧 {i}：r 必须 > 0，得到 {k.r}")
        if not 0 < k.fov < 180:
            raise ValueError(f"关键帧 {i}：fov 必须在 (0, 180)，得到 {k.fov}")
        if not all(SUBJECT_UV_MIN <= c <= SUBJECT_UV_MAX for c in k.subject_uv):
            raise ValueError(f"关键帧 {i}：subject_uv 必须在 [{SUBJECT_UV_MIN:g}, {SUBJECT_UV_MAX:g}]，"
                             f"得到 {k.subject_uv}")
    tm = timing
    for name in ("duration", "path_smoothing", "near_weight", "near_height_floor", "progress_smoothing",
                 "ramp_in", "ramp_out", "speed_start", "speed_end"):
        if not math.isfinite(getattr(tm, name)):
            raise ValueError(f"{name} 必须是有限数，得到 {getattr(tm, name)}")
    if not tm.duration >= MIN_DURATION:
        raise ValueError(f"duration 必须 ≥ {MIN_DURATION:.4f} s（节奏曲线至少 3 个时间步），得到 {tm.duration}")
    for name in ("path_smoothing", "progress_smoothing", "near_weight", "ramp_in", "ramp_out"):
        if getattr(tm, name) < 0:
            raise ValueError(f"{name} 必须 ≥ 0，得到 {getattr(tm, name)}")
    if not tm.near_height_floor > 0:
        raise ValueError(f"near_height_floor 必须 > 0，得到 {tm.near_height_floor}")
    if tm.ramp_in + tm.ramp_out > tm.duration:
        raise ValueError("ramp_in + ramp_out 不能超过 duration")
    if not (0 < tm.speed_start <= 1 and 0 < tm.speed_end <= 1):
        raise ValueError("speed_start / speed_end 必须在 (0, 1]")


def optional_block(spec: dict, name: str) -> dict:
    """取路径文件中的可选块（省略时为空字典）。

    Args:
        spec: 路径文件内容（JSON 解析后的字典）。
        name: 块名：`progress`、`rhythm`（本模块）或 `exposure`、`fade`（`exposure_curve`）。

    Returns:
        该块的字典。

    Raises:
        ValueError: 该块存在但不是 JSON 对象。
    """
    block = spec.get(name, {})
    if not isinstance(block, dict):
        raise ValueError(f"{name} 必须是 JSON 对象，得到 {block!r}")
    return block


def load_camera_path(spec: dict) -> CameraPath:
    """由路径文件内容（JSON 解析后的字典）构造 `CameraPath`。

    Args:
        spec: 含 `duration`、`keyframes`，可选 `path_smoothing`、`progress`、`rhythm` 块；
            格式见 `docs/plans/v2_camera_path_plan.md` §5.1。省略的可选项取 `PathTiming` 默认值。

    Returns:
        `CameraPath`。

    Raises:
        ValueError: 缺少必填字段、结构错误（块不是对象、`keyframes` 不是数组、`subject_uv` 不是两个数）或取值非法。
    """
    try:
        progress = optional_block(spec, "progress")
        rhythm = optional_block(spec, "rhythm")
        defaults = PathTiming(duration=float(spec["duration"]))
        timing = PathTiming(
            duration=defaults.duration,
            path_smoothing=float(spec.get("path_smoothing", defaults.path_smoothing)),
            near_weight=float(progress.get("near_weight", defaults.near_weight)),
            near_height_floor=float(progress.get("near_height_floor", defaults.near_height_floor)),
            progress_smoothing=float(progress.get("smoothing", defaults.progress_smoothing)),
            ramp_in=float(rhythm.get("ramp_in", defaults.ramp_in)),
            ramp_out=float(rhythm.get("ramp_out", defaults.ramp_out)),
            speed_start=float(rhythm.get("speed_start", defaults.speed_start)),
            speed_end=float(rhythm.get("speed_end", defaults.speed_end)),
        )
        if not isinstance(spec["keyframes"], list):
            raise ValueError(f"keyframes 必须是数组，得到 {spec['keyframes']!r}")
        keyframes = []
        for i, k in enumerate(spec["keyframes"]):
            uv = k["subject_uv"]
            if not isinstance(uv, list) or len(uv) != 2:
                raise ValueError(f"关键帧 {i}：subject_uv 必须是两个数 [u, v]，得到 {uv!r}")
            keyframes.append(Keyframe(r=float(k["r"]), phi_deg=float(k["phi_deg"]), z=float(k["z"]),
                                      fov=float(k["fov"]), roll=float(k["roll"]), subject_uv=(float(uv[0]), float(uv[1])),
                                      label=str(k.get("label", ""))))
    except KeyError as exc:
        raise ValueError(f"路径文件缺少必填字段 {exc}") from exc
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"路径文件结构错误：{exc}") from exc
    return CameraPath(keyframes, timing)
