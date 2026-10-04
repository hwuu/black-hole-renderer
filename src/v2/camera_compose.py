"""运镜的数值工具：保形插值、高斯平滑、相机基向量与构图求解。

这些函数只做几何与插值运算，不依赖 Taichi，供 `src/v2/camera_path.py` 使用。
全部函数支持批量输入（首维为样本），单个样本时传形状为 `(1, ·)` 的数组即可。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# 构图求解：不动点迭代次数（常见构图 3–5 次收敛）、收敛判据（画面坐标误差）与阻尼高斯–牛顿的最大迭代次数
COMPOSE_ITERATIONS = 30
COMPOSE_TOLERANCE = 1e-6
COMPOSE_REFINE_ITERATIONS = 100
_WORLD_UP = np.array([0.0, 0.0, 1.0])


def pchip(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson 单调保形三次插值，两端导数取 0。

    Args:
        x: 节点横坐标，形状 `(K,)`，严格递增，K ≥ 2。
        y: 节点函数值，形状 `(K,)`。
        xq: 查询点，任意形状；超出 `[x_0, x_{K−1}]` 时按端点区间外推（调用方应保证在区间内）。

    Returns:
        与 `xq` 同形状的插值结果；节点处精确等于 `y`，相邻节点之间不越出两端值的范围。

    Formula:
        割线斜率 d_k = (y_{k+1} − y_k) / h_k，h_k = x_{k+1} − x_k；
        内部节点导数：d_{k−1}·d_k > 0 时取加权调和平均
            m_k = (w₁ + w₂) / (w₁ / d_{k−1} + w₂ / d_k)，w₁ = 2h_k + h_{k−1}，w₂ = h_k + 2h_{k−1}，
        否则（局部极值或平台）m_k = 0；两端 m_0 = m_{K−1} = 0。
        区间内用三次 Hermite 基：y = h₀₀·y_i + h₁₀·h_i·m_i + h₀₁·y_{i+1} + h₁₁·h_i·m_{i+1}，
        s = (x − x_i) / h_i，h₀₀ = 2s³ − 3s² + 1，h₁₀ = s³ − 2s² + s，h₀₁ = −2s³ + 3s²，h₁₁ = s³ − s²。

    Simplifications:
        一阶导数连续（C¹），二阶导数在节点处一般不连续；运镜中由 `gaussian_smooth` 消除。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    for k in range(1, len(x) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    i = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    s = (xq - x[i]) / h[i]
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1]


def gaussian_smooth(values: np.ndarray, sigma_samples: float) -> np.ndarray:
    """沿首维做截断高斯平滑，两端按端点值延拓。

    Args:
        values: 均匀网格上的采样，形状 `(N,)` 或 `(N, C)`（逐列平滑）。
        sigma_samples: 高斯标准差（以网格点数计，≥ 0）；0 时原样返回副本。

    Returns:
        与 `values` 同形状的平滑结果。

    Formula:
        ṽ_j = Σ_{k=−n}^{n} v_{j+k}·G_k / Σ G_k，G_k = exp(−k² / (2σ²))，n = ⌊3σ⌋；
        越界的 v 取端点值。

    Simplifications:
        截断到 ±3σ（舍去的权重约 0.27%）；端点延拓使两端附近的平滑偏向端点值。
    """
    arr = np.asarray(values, dtype=np.float64)
    if sigma_samples <= 0:
        return arr.copy()
    n = int(3 * sigma_samples)
    kernel = np.exp(-0.5 * (np.arange(-n, n + 1) / sigma_samples) ** 2)
    kernel /= kernel.sum()
    cols = arr.reshape(len(arr), -1)
    out = np.stack([np.convolve(np.pad(c, n, mode="edge"), kernel, mode="valid") for c in cols.T], axis=1)
    return out.reshape(arr.shape)


class UniformSpline:
    """均匀网格上的三次样条：把网格采样还原为二阶导数连续的函数。

    Args:
        x0: 网格起点。
        step: 网格间距（> 0）。
        values: 网格采样，形状 `(N,)` 或 `(N, C)`（逐列独立插值），N ≥ 4。

    Formula:
        节点二阶导数 M 满足 M_{i−1} + 4M_i + M_{i+1} = 6(y_{i−1} − 2y_i + y_{i+1}) / h²（i = 1 … N−2），
        端点条件 M_0 = 2M_1 − M_2、M_{N−1} = 2M_{N−2} − M_{N−3}（二阶导数在首末区间线性外推），
        代入后首末行变为 6M_1 = rhs_1、6M_{N−2} = rhs_{N−2}，用追赶法求解；区间内
        y(x) = (1 − s)·y_i + s·y_{i+1} + h²/6·[((1 − s)³ − (1 − s))·M_i + (s³ − s)·M_{i+1}]，s = (x − x_i) / h。

    Simplifications:
        函数值、一阶与二阶导数连续，三阶导数在节点处可有跳变；端点二阶导数由相邻区间线性外推，
        避免自然样条（端点二阶导数为 0）在首末区间造成的三阶导数尖峰。
    """

    def __init__(self, x0: float, step: float, values: np.ndarray):
        """求解节点二阶导数 M（见类说明的 Formula）。

        Args:
            x0: 网格起点。
            step: 网格间距（> 0）。
            values: 网格采样，形状 `(N,)` 或 `(N, C)`，N ≥ 4。
        """
        self.x0, self.h = float(x0), float(step)
        self.y = np.asarray(values, dtype=np.float64)
        y2 = self.y.reshape(len(self.y), -1)
        n = len(y2)
        rhs = 6.0 * (y2[:-2] - 2.0 * y2[1:-1] + y2[2:]) / self.h**2
        # 追赶法（Thomas）求 M_1 … M_{N−2}：三对角 (1, 4, 1)，代入端点条件后首末行变为对角元 6、非对角元 0
        diag = np.full(n - 2, 4.0)
        diag[0] = diag[-1] = 6.0
        lower = np.ones(n - 2)
        lower[-1] = 0.0
        upper = np.ones(n - 2)
        upper[0] = 0.0
        c = np.zeros(n - 2)
        d = np.zeros_like(rhs)
        c[0], d[0] = upper[0] / diag[0], rhs[0] / diag[0]
        for i in range(1, n - 2):
            denom = diag[i] - lower[i] * c[i - 1]
            c[i] = upper[i] / denom
            d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom
        m = np.zeros_like(y2)
        m[n - 2] = d[-1]
        for i in range(n - 4, -1, -1):
            m[i + 1] = d[i] - c[i] * m[i + 2]
        m[0] = 2 * m[1] - m[2]
        m[-1] = 2 * m[-2] - m[-3]
        self.m = m.reshape(self.y.shape)

    def _locate(self, xq) -> Tuple[np.ndarray, np.ndarray]:
        """查询点所在区间序号与区间内归一化位置。

        Args:
            xq: 查询点，标量或 `(Q,)` 数组；截断到网格范围 `[x0, x0 + (N − 1)·h]`。

        Returns:
            `(i, s)`：区间序号（整数，0 … N−2）与 s = (x − x_i) / h ∈ [0, 1]；多列采样时 s 末尾补一维以便广播。
        """
        n = len(self.y)
        u = np.clip((np.asarray(xq, dtype=np.float64) - self.x0) / self.h, 0.0, n - 1.0)
        i = np.minimum(u.astype(int), n - 2)
        s = u - i
        return i, (s[..., None] if self.y.ndim == 2 else s)

    def __call__(self, xq) -> np.ndarray:
        """样条在查询点处的函数值。

        Args:
            xq: 查询点，标量或 `(Q,)` 数组；超出网格范围时取端点值。

        Returns:
            标量查询时形状同 `values[0]`（标量或 `(C,)`）；数组查询时形状为 `(Q,)` 或 `(Q, C)`。
        """
        i, s = self._locate(xq)
        a = 1.0 - s
        return (a * self.y[i] + s * self.y[i + 1]
                + self.h**2 / 6.0 * ((a**3 - a) * self.m[i] + (s**3 - s) * self.m[i + 1]))

    def derivative(self, xq) -> np.ndarray:
        """样条在查询点处的一阶导数 dy/dx。

        Args:
            xq: 查询点，标量或 `(Q,)` 数组；超出网格范围时取端点区间的端点导数。

        Returns:
            形状规则同 `__call__`。

        Formula:
            y′ = (y_{i+1} − y_i) / h + h/6·[−(3(1 − s)² − 1)·M_i + (3s² − 1)·M_{i+1}]
        """
        i, s = self._locate(xq)
        a = 1.0 - s
        return ((self.y[i + 1] - self.y[i]) / self.h
                + self.h / 6.0 * (-(3 * a**2 - 1) * self.m[i] + (3 * s**2 - 1) * self.m[i + 1]))


def camera_basis(forward: np.ndarray, roll_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """由光轴与滚转角构造画面水平 / 竖直方向（与 `build_camera_v1_compatible` 同一构造）。

    Args:
        forward: 单位光轴方向，形状 `(N, 3)`。
        roll_deg: 滚转角 ρ（度），形状 `(N,)`；正值使画面左低右高。

    Returns:
        `(right, up)`：形状均为 `(N, 3)` 的单位向量，与 forward 构成正交基。

    Formula:
        r = normalize(f × ẑ)，u = normalize(r × f)；
        r' = cos ρ · r − sin ρ · u，u' = sin ρ · r + cos ρ · u。
        f 与 ẑ 平行时 r 取 +x（与 `build_camera_v1_compatible` 相同的退化处理）。
    """
    right = np.cross(forward, _WORLD_UP)
    norm = np.linalg.norm(right, axis=1, keepdims=True)
    right = np.where(norm < 1e-6, np.array([1.0, 0.0, 0.0]), right / np.maximum(norm, 1e-300))
    up = np.cross(right, forward)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    a = np.radians(roll_deg)[:, None]
    return np.cos(a) * right - np.sin(a) * up, np.sin(a) * right + np.cos(a) * up


def _rotate(vec: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues 旋转：把向量绕单位轴旋转给定角度。

    Args:
        vec: 待旋转向量，形状 `(N, 3)`。
        axis: 单位旋转轴，形状 `(N, 3)`。
        angle: 旋转角（弧度，右手定则），形状 `(N,)`。

    Returns:
        旋转后的向量，形状 `(N, 3)`。

    Formula:
        v' = v·cos θ + (k × v)·sin θ + k·(k·v)·(1 − cos θ)
    """
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    kv = np.sum(axis * vec, axis=1, keepdims=True)
    return vec * c + np.cross(axis, vec) * s + axis * kv * (1 - c)


def solve_forward(pos: np.ndarray, subject_uv: np.ndarray, fov_deg: np.ndarray, aspect: float,
                  roll_deg: np.ndarray) -> np.ndarray:
    """求相机光轴，使黑洞（原点）投影到画面上的指定位置。

    Args:
        pos: 相机位置（r_s），形状 `(N, 3)`，不能为原点。
        subject_uv: 黑洞的目标画面位置 `(u, v)`，形状 `(N, 2)`；u 从左到右、v 从上到下，[0, 1]。
        fov_deg: 竖直视野角（度），形状 `(N,)`。
        aspect: 画面宽高比 W / H。
        roll_deg: 滚转角（度），形状 `(N,)`。

    Returns:
        单位光轴 forward，形状 `(N, 3)`；以它和 `roll_deg` 构造的针孔相机把原点投影到 `(u, v)`。

    Formula:
        目标在相机系中的方向 d_cam ∝ (x, y, 1)，x = (u − ½)·W_p，y = (½ − v)·H_p，
        H_p = 2·tan(fov / 2)，W_p = aspect·H_p；
        f₀ = ŝ = −p / |p|；第 k 步由 (f_k, ρ) 得 (r_k, u_k)，d_k = normalize(f_k + x·r_k + y·u_k)，
        把 f_k 绕 normalize(d_k × ŝ) 旋转 ∠(d_k, ŝ) 得 f_{k+1}（d 随之转到 ŝ 附近）。

        不动点迭代 `COMPOSE_ITERATIONS` 次后检查投影误差；误差超过 `COMPOSE_TOLERANCE` 的样本（光轴接近竖直时，
        画面基随光轴方位变化剧烈，不动点迭代可能不收敛）改用 `_refine_forward` 的阻尼高斯–牛顿迭代。

    Physical Meaning:
        史瓦西黑洞的阴影中心严格位于原点方向（球对称），所以"黑洞在画面中的位置"就是原点方向的
        针孔投影；求出的 forward 把黑洞放到构图位置（如三分线）。

    Simplifications:
        不检查目标点是否在画面外。

    Raises:
        ValueError: 阻尼高斯–牛顿迭代后投影误差仍超过 `COMPOSE_TOLERANCE`（该构图在此相机位置无解，见 `_refine_forward`）。
    """
    pos = np.asarray(pos, dtype=np.float64)
    s = -pos / np.linalg.norm(pos, axis=1, keepdims=True)
    hp = 2 * np.tan(np.radians(fov_deg) / 2)
    x = ((subject_uv[:, 0] - 0.5) * hp * aspect)[:, None]
    y = ((0.5 - subject_uv[:, 1]) * hp)[:, None]
    f = s.copy()
    for _ in range(COMPOSE_ITERATIONS):
        right, up = camera_basis(f, roll_deg)
        d = f + x * right + y * up
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        axis = np.cross(d, s)
        sn = np.linalg.norm(axis, axis=1)
        safe = sn > 1e-12
        axis = np.where(safe[:, None], axis / np.maximum(sn, 1e-300)[:, None], 0.0)
        angle = np.where(safe, np.arctan2(sn, np.sum(d * s, axis=1)), 0.0)
        f = _rotate(f, axis, angle)
        f /= np.linalg.norm(f, axis=1, keepdims=True)
    err = np.abs(project_origin(pos, f, roll_deg, fov_deg, aspect) - subject_uv).max(axis=1)
    for i in np.nonzero(~(err <= COMPOSE_TOLERANCE))[0]:
        f[i] = _refine_forward(pos[i], subject_uv[i], float(fov_deg[i]), aspect, float(roll_deg[i]), f[i])
    return f


def _refine_forward(pos: np.ndarray, uv: np.ndarray, fov_deg: float, aspect: float, roll_deg: float,
                    f0: np.ndarray) -> np.ndarray:
    """阻尼高斯–牛顿迭代：在光轴的切平面上求解"原点投影 = 目标画面位置"。

    Args:
        pos: 相机位置（r_s），形状 `(3,)`。
        uv: 目标画面位置 `(u, v)`，形状 `(2,)`。
        fov_deg: 竖直视野角（度）。
        aspect: 画面宽高比 W / H。
        roll_deg: 滚转角（度）。
        f0: 初始单位光轴，形状 `(3,)`。

    Returns:
        单位光轴，形状 `(3,)`，原点投影误差 ≤ `COMPOSE_TOLERANCE`。

    Formula:
        每步以当前光轴 f 为基点取切平面正交基 (e₁, e₂)，f(q) = normalize(f + q₁·e₁ + q₂·e₂)；
        残差 e(q) = proj(f(q)) − uv，J 由中心差分（步长 1e-6）得到；
        Δq = −(JᵀJ + λI)⁻¹ Jᵀ e，残差下降则接受（f ← f(Δq)）并 λ ← λ/3，否则 λ ← 3λ。

    Simplifications:
        切平面参数化在任何光轴方向都没有坐标奇点（方位角 / 仰角参数化在光轴竖直时退化）。

    Raises:
        ValueError: 迭代 `COMPOSE_REFINE_ITERATIONS` 次后仍未收敛。黑洞方向离竖直方向只有几度时，
            由于画面基由世界上方向 +z 决定，可达的画面位置只是过中心的一条窄带，此时构图确实无解。
    """
    def project(f):
        return project_origin(pos[None, :], f[None, :], np.array([roll_deg]), np.array([fov_deg]), aspect)[0]

    f = f0 / np.linalg.norm(f0)
    e = project(f) - uv
    lam, h = 1e-3, 1e-6
    for _ in range(COMPOSE_REFINE_ITERATIONS):
        if np.abs(e).max() <= COMPOSE_TOLERANCE:
            return f
        helper = np.array([1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(f, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(f, e1)

        def moved(q):
            g = f + q[0] * e1 + q[1] * e2
            return g / np.linalg.norm(g)

        jac = np.stack([(project(moved(h * dq)) - project(moved(-h * dq))) / (2 * h) for dq in np.eye(2)], axis=1)
        step = -np.linalg.solve(jac.T @ jac + lam * np.eye(2), jac.T @ e)
        f_new = moved(step)
        e_new = project(f_new) - uv
        if np.abs(e_new).max() < np.abs(e).max():
            f, e, lam = f_new, e_new, lam / 3.0
        else:
            lam *= 3.0
    if np.abs(e).max() <= COMPOSE_TOLERANCE:
        return f
    raise ValueError(f"构图无解：相机位置 {pos.tolist()} 处无法把黑洞放到画面位置 {uv.tolist()}（最小误差 "
                     f"{np.abs(e).max():.2e}）。黑洞方向接近竖直时，画面水平方向由世界上方向 +z 决定，黑洞只能落在"
                     "过画面中心的一条直线附近；请降低相机高度或把目标位置移近画面中心")


def project_origin(pos: np.ndarray, forward: np.ndarray, roll_deg: np.ndarray, fov_deg: np.ndarray,
                   aspect: float) -> np.ndarray:
    """计算原点（黑洞）在画面中的投影位置。

    Args:
        pos: 相机位置（r_s），形状 `(N, 3)`。
        forward: 单位光轴，形状 `(N, 3)`。
        roll_deg: 滚转角（度），形状 `(N,)`。
        fov_deg: 竖直视野角（度），形状 `(N,)`。
        aspect: 画面宽高比 W / H。

    Returns:
        画面坐标 `(u, v)`，形状 `(N, 2)`；原点在相机后方时结果无意义（调用方保证在前方）。

    Formula:
        ŝ = −p / |p|，u = (ŝ·r) / (ŝ·f) / W_p + ½，v = ½ − (ŝ·u) / (ŝ·f) / H_p。
    """
    s = -pos / np.linalg.norm(pos, axis=1, keepdims=True)
    right, up = camera_basis(forward, roll_deg)
    hp = 2 * np.tan(np.radians(fov_deg) / 2)
    depth = np.sum(s * forward, axis=1)
    u = np.sum(s * right, axis=1) / depth / (hp * aspect) + 0.5
    v = 0.5 - np.sum(s * up, axis=1) / depth / hp
    return np.stack([u, v], axis=1)
