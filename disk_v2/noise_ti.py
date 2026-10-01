"""Disk V2 程序化噪声库（v2.3 S3）。

提供体积密度场的全部噪声原语（Taichi `@ti.func`）：

- 整数哈希（lowbias32 变体）与 `[0, 1)` 浮点哈希。
- `gnoise`：3D 梯度噪声，y 方向按整数周期回绕（φ 无缝）。
- `vnoise`：3D value noise，y 方向同样周期回绕。
- `softplus`：数值安全的 `ln(1 + eˣ)`。
- `cascade`：乘性级联噪声（倍率 3、等权、支持小数八度）+ softplus 软阈值。
- `fbm_gradient`：梯度噪声 fBm（可配八度数与增益）。

全部为纯函数（无隐藏状态、无随机种子依赖），同输入必同输出；
NumPy 镜像（`tests/unit/test_disk_v2_noise_ti.py` 内）逐位对齐做 parity。

坐标约定：调用方传"拉格朗日流坐标"，噪声本身不感知时间/旋转
（平流由 `disk_v2/advection.py` 负责）；y 方向周期必须为正整数，
否则无缝性不成立（由 `cascade`/`fbm_gradient` 内部按八度翻倍保持）。
"""

import taichi as ti


@ti.func
def hash_u32(x):
    """32 位整数哈希（lowbias32 变体）。

    Args:
        x: `u32` 标量。

    Returns:
        `u32` 哈希值。同输入必同输出；输出在 `u32` 上均匀性良好
        （低偏置：输出每一位对输入每一位都敏感）。
    """
    x = x ^ (x >> ti.u32(16))
    x = x * ti.u32(0x7FEB352D)
    x = x ^ (x >> ti.u32(15))
    x = x * ti.u32(0x2C1B3C6D)
    x = x ^ (x >> ti.u32(16))
    return x


@ti.func
def hash3(ix, iy, iz):
    """三维整数格点哈希。

    Args:
        ix, iy, iz: `i32` 格点坐标（可为负，内部偏移 +4096 后哈希）。

    Returns:
        `u32` 哈希值。三轴各自混入不同素数，保证轴向独立。
    """
    h = hash_u32(ti.cast(ix + 4096, ti.u32) * ti.u32(1597334677))
    h = hash_u32(h ^ (ti.cast(iy + 4096, ti.u32) * ti.u32(1103515245)))
    h = hash_u32(h ^ (ti.cast(iz + 4096, ti.u32) * ti.u32(1234567891)))
    return h


@ti.func
def hashf(a, b, c):
    """三维整数哈希 → `[0, 1)` 浮点值。

    Args:
        a, b, c: `i32` 标量。

    Returns:
        取 `hash3` 低 24 位除以 `2^24`，分辨率约 6e-8，
        足够 value noise 插值使用。
    """
    return ti.cast(hash3(a, b, c) & ti.u32(0xFFFFFF), ti.f32) / 16777216.0


@ti.func
def _fade(t):
    """Perlin 五次平滑（梯度噪声插值权）。

    Args:
        t: 格内偏移，`[0, 1]`。

    Returns:
        插值权 `6t⁵ − 15t⁴ + 10t³`，`[0, 1]`，两端一阶、二阶导为 0。
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@ti.func
def _smooth3(t):
    """三次平滑（value noise 插值权）。

    Args:
        t: 格内偏移，`[0, 1]`。

    Returns:
        插值权 `3t² − 2t³`，`[0, 1]`，两端一阶导为 0（与参考实现 vnoise 一致）。
    """
    return t * t * (3.0 - 2.0 * t)


@ti.func
def _corner(xi, yi, zi, dx, dy, dz, period):
    """梯度噪声格点贡献：`hash` 派生梯度 · 偏移向量。

    Args:
        xi, yi, zi: `i32` 格点坐标。
        dx, dy, dz: 采样点到该格点的偏移（格距单位）。
        period: y 方向周期（正整数）；`yi` 先回绕再哈希。

    Returns:
        标量贡献 `gx·dx + gy·dy + gz·dz`，梯度分量约在 `[-1, 1]`。

    Notes:
        梯度取 `hash3` 的三个 10 位段，`/511.5 − 1` 映射到 `[-1, 1]`。
    """
    yy = ((yi % period) + period) % period
    h = hash3(xi, yy, zi)
    gx = ti.cast(h & ti.u32(1023), ti.f32) / 511.5 - 1.0
    gy = ti.cast((h >> ti.u32(10)) & ti.u32(1023), ti.f32) / 511.5 - 1.0
    gz = ti.cast((h >> ti.u32(20)) & ti.u32(1023), ti.f32) / 511.5 - 1.0
    return gx * dx + gy * dy + gz * dz


@ti.func
def gnoise(x, y, z, period):
    """3D 梯度噪声，y 方向周期 `period`（φ 无缝）。

    Args:
        x, y, z: 连续坐标；y 以 `period`（正整数）为周期回绕。
        period: y 方向周期，正整数。

    Returns:
        标量，约 `[-1, 1]`，零均值（f32 实测 std ≈ 0.19；调用方按实测标定归一）。

    Formula:
        8 个格点梯度贡献经 `_fade` 三线性插值（Perlin 1985 改进版）。
    """
    fx0 = ti.floor(x)
    fy0 = ti.floor(y)
    fz0 = ti.floor(z)
    xi = ti.cast(fx0, ti.i32)
    yi = ti.cast(fy0, ti.i32)
    zi = ti.cast(fz0, ti.i32)
    fx = x - fx0
    fy = y - fy0
    fz = z - fz0
    ux = _fade(fx)
    uy = _fade(fy)
    uz = _fade(fz)
    n000 = _corner(xi, yi, zi, fx, fy, fz, period)
    n100 = _corner(xi + 1, yi, zi, fx - 1.0, fy, fz, period)
    n010 = _corner(xi, yi + 1, zi, fx, fy - 1.0, fz, period)
    n110 = _corner(xi + 1, yi + 1, zi, fx - 1.0, fy - 1.0, fz, period)
    n001 = _corner(xi, yi, zi + 1, fx, fy, fz - 1.0, period)
    n101 = _corner(xi + 1, yi, zi + 1, fx - 1.0, fy, fz - 1.0, period)
    n011 = _corner(xi, yi + 1, zi + 1, fx, fy - 1.0, fz - 1.0, period)
    n111 = _corner(xi + 1, yi + 1, zi + 1, fx - 1.0, fy - 1.0, fz - 1.0, period)
    nx00 = n000 + ux * (n100 - n000)
    nx10 = n010 + ux * (n110 - n010)
    nx01 = n001 + ux * (n101 - n001)
    nx11 = n011 + ux * (n111 - n011)
    nxy0 = nx00 + uy * (nx10 - nx00)
    nxy1 = nx01 + uy * (nx11 - nx01)
    return nxy0 + uz * (nxy1 - nxy0)


@ti.func
def vnoise(x, y, z, period):
    """3D value noise，y 方向周期 `period`（φ 无缝）。

    Args:
        x, y, z: 连续坐标；y 以 `period`（正整数）为周期回绕。
        period: y 方向周期，正整数。

    Returns:
        标量，`[-1, 1]`，零均值（f32 实测 std ≈ 0.31；调用方按实测标定归一）。

    Formula:
        8 个格点随机值（`hashf`）经 `_smooth3` 三线性插值。
    """
    fx0 = ti.floor(x)
    fy0 = ti.floor(y)
    fz0 = ti.floor(z)
    xi = ti.cast(fx0, ti.i32)
    yi = ti.cast(fy0, ti.i32)
    zi = ti.cast(fz0, ti.i32)
    fx = x - fx0
    fy = y - fy0
    fz = z - fz0
    ux = _smooth3(fx)
    uy = _smooth3(fy)
    uz = _smooth3(fz)
    y0 = ((yi % period) + period) % period
    y1 = (((yi + 1) % period) + period) % period
    v000 = hashf(xi, y0, zi) * 2.0 - 1.0
    v100 = hashf(xi + 1, y0, zi) * 2.0 - 1.0
    v010 = hashf(xi, y1, zi) * 2.0 - 1.0
    v110 = hashf(xi + 1, y1, zi) * 2.0 - 1.0
    v001 = hashf(xi, y0, zi + 1) * 2.0 - 1.0
    v101 = hashf(xi + 1, y0, zi + 1) * 2.0 - 1.0
    v011 = hashf(xi, y1, zi + 1) * 2.0 - 1.0
    v111 = hashf(xi + 1, y1, zi + 1) * 2.0 - 1.0
    a0 = v000 + ux * (v100 - v000)
    a1 = v010 + ux * (v110 - v010)
    a2 = v001 + ux * (v101 - v001)
    a3 = v011 + ux * (v111 - v011)
    b0 = a0 + uy * (a1 - a0)
    b1 = a2 + uy * (a3 - a2)
    return b0 + uz * (b1 - b0)


@ti.func
def softplus(x):
    """数值安全的 `ln(1 + eˣ)`。

    Args:
        x: 标量。

    Returns:
        `x ≥ 20` 时直接返回 `x`（相对误差 < 2e-9），否则精确计算。
    """
    out = x
    if x < 20.0:
        out = ti.log(1.0 + ti.exp(x))
    return out


@ti.func
def cascade(x, y, z, per_y, l0, l1, con):
    """乘性级联噪声（倍率 3、八度等权、小数八度）+ softplus 软阈值。

    Args:
        x, y, z: 连续坐标；y 以 `per_y·3^l` 为周期（每八度自动翻三倍）。
        per_y: y 方向基频周期，正整数。
        l0, l1: 八度区间 `[l0, l1)`，可为小数（权重按覆盖长度线性取值，
            整数边界处连续）。
        con: 软阈值对比度（越大暗缝越深、亮丝越锐）。

    Returns:
        标量，≥ 0。大部分区域接近 0（暗缝），少数区域近似线性（亮丝）。

    Formula:
        ```
        S   = Σ_l w_l · ln(1 + 0.1·n_l)，w_l = clamp(min(l1, l+1) − max(l0, l), 0, 1)
        out = softplus(con · S) = ln(1 + (Π_l (1 + 0.1·w_l·n_l))^con)
        ```
        `n_l = vnoise(3^l·p, per_y·3^l)`；倍率 3 + 等权 → 大小尺度对比相当。

    Physical Meaning:
        湍流密度场的"暗背景 + 锐亮丝"一维化近似：乘性级联使任一八度的
        低值都会整段压暗（暗缝），softplus 软阈值替代硬 clip 避免阶梯。

    Simplifications:
        - 内部最多迭代 5 个整数八度（`l1 − l0 ≤ 5` 约定由调用方保证）。
        - 八度增益固定 1（等权），不做增益参数化。
    """
    s = 0.0
    i0 = ti.cast(ti.floor(l0), ti.i32)
    for k in range(5):
        lv = i0 + k
        lvf = ti.cast(lv, ti.f32)
        w = ti.min(ti.max(ti.min(l1, lvf + 1.0) - ti.max(l0, lvf), 0.0), 1.0)
        if w > 0.0 and lv >= 0:
            f = 1.0
            per = per_y
            for _q in range(lv):
                f *= 3.0
                per *= 3
            n = vnoise(x * f, y * f, z * f, per)
            s += ti.log(1.0 + 0.1 * n * w)
    return softplus(con * s)


@ti.func
def fbm_gradient(x, y, z, period, octaves: ti.template(), gain):
    """梯度噪声 fBm（固定八度数，编译期展开）。

    Args:
        x, y, z: 连续坐标；y 以 `period·2^o` 为周期（每八度自动翻倍）。
        period: y 方向基频周期，正整数。
        octaves: 八度数（`ti.template()`，编译期常量，1 ~ 5）。
        gain: 每八度振幅比（原型主涨落 0.5，低频层 0.5）。

    Returns:
        标量，零均值；原始和未归一（调用方按实测 std 归一到单位方差）。

    Formula:
        ```
        f = Σ_o gain^o · gnoise(2^o·p, period·2^o)，o = 0..octaves−1
        ```
    """
    s = 0.0
    for o in ti.static(range(octaves)):
        f = 2 ** o
        s += gain ** o * gnoise(x * f, y * f, z * f, period * f)
    return s
