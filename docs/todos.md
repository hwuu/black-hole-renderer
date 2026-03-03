# TODO List

## 吸积盘纹理缓存

### 需求
吸积盘渲染太花时间（~30-130秒），添加缓存机制：
1. 基于 CLI 参数（ar1, ar2, seed）生成缓存 key
2. 参数相同且缓存命中时直接加载，避免重复生成
3. 新增 `--force_rerender_accretion_disk` 开关，强制重新渲染并更新缓存

### 设计

**缓存目录**: `output/.accretion_disk_cache/`

**缓存 key 生成**:
```python
def get_disk_cache_key(r_inner, r_outer, seed, enable_rt=True):
    return f"disk_{r_inner:.2f}_{r_outer:.2f}_{seed}_{enable_rt}.npy"
```

**缓存查找逻辑**:
```python
def load_cached_disk_texture(r_inner, r_outer, seed, enable_rt=True, force=False):
    cache_dir = "output/accretion_disk_cache"
    cache_key = get_disk_cache_key(r_inner, r_outer, seed, enable_rt)
    cache_path = os.path.join(cache_dir, cache_key)
    
    if not force and os.path.exists(cache_path):
        print(f"Loading cached disk texture: {cache_key}")
        return np.load(cache_path)
    
    # 生成并缓存
    tex = generate_disk_texture(r_inner=r_inner, r_outer=r_outer, seed=seed, enable_rt=enable_rt)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, tex)
    return tex
```

**CLI 参数**:
- `--force_rerender_accretion_disk`: 强制重新生成吸积盘纹理（默认 False）

### 实施步骤

1. [x] 在 `generate_disk_texture` 附近添加 `load_cached_disk_texture` 函数
2. [x] 在 CLI 参数中添加 `--force_rerender_accretion_disk`
3. [x] 修改 `render_image` 函数使用缓存加载
4. [x] 验证缓存命中/未命中逻辑正确

---

## 吸积盘结构湍流扰动 + 温度调整

### 问题

1. **温度偏低**：用外部纹理渲染时，吸积环整体温度偏低
2. **结构太光滑**：filament、hotspot、arms 的边缘是光滑的高斯，缺少湍流扰动

### 当前代码分析

```
密度场:
  density = base + spiral + turbulence + hotspot + arcs + rt_spikes
  
温度场:
  temp_struct = sum(结构 × 温度增量)
  temperature_field = max(temp_base, temp_struct_scaled)
  
最终输出:
  RGB = 黑体色 × sqrt(temperature)
  Alpha = density
```

**问题所在**：
- `turbulence` 已是噪声场，但 `spiral/arc/hotspot` 结构本身是光滑高斯
- 结构的**密度轮廓**光滑，**温度增量**也光滑
- 物理上，湍流应该同时扰动密度分布和温度分布

### 方案

#### 1. 温度调整

| 参数 | 当前值 | 目标值 | 说明 |
|------|--------|--------|------|
| T_max | 7000K | 10000K | 提高高温端上限 |
| density base | 0.05 | 0.10 | 增加基础密度 |

#### 2. 湍流扰动实现

**方法 A：强度调制**（推荐，实现简单）

```python
# 生成扰动噪声（与结构同频率）
disturb_noise = _fbm_noise((n_r, n_phi), rng, octaves=3, persistence=0.5, base_scale=8, wrap_u=True)
disturb_mod = 0.4 + 0.6 * disturb_noise  # 0.4-1.0

# 对密度场调制
density = (base + 0.22*spiral + ...) * disturb_mod

# 对温度场调制
temp_struct = temp_struct * disturb_mod
```

**方法 B：坐标扰动**（更物理，实现复杂）

用噪声偏移采样坐标，使结构边缘被"拉扯"成不规则形状。暂不采用，因为：
- 实现复杂度高
- 需要仔细处理边界条件
- 可能引入采样伪影

### 实施步骤

1. [x] 分析当前代码结构
2. [x] 调整温度参数（T_max → 10000K，density base → 0.15）
3. [x] 生成扰动噪声 disturb_noise（多频 + 剪切，借鉴 turbulence 结构）
4. [x] 对 density 和 temp_struct 应用扰动调制
5. [x] 运行 check_texture.py 验证效果
6. [x] 用外部纹理渲染验证整体效果

### 预期效果

- 结构边缘不再光滑，呈现絮状破碎感
- 整体温度提升，视觉效果更亮更热
- 保持物理合理性（湍流同时影响密度和温度）

---

## 镜头光晕/光斑效果

### 背景
黑洞吸积盘是强光源，应该有镜头光学效果（类似 interstellar 电影）。

### 方案

#### 1. Bloom（泛光）
提取高亮区域 → 高斯模糊 → 叠加回原图
```python
bright = threshold(image)
bloom = gaussian_blur(bright, sigma)
result = image + bloom * intensity
```

**实现方式**：
- `scipy.ndimage.gaussian_filter`
- 或 Taichi 并行实现

#### 2. Lens Flare（镜头光斑）

**Sprite-based 方法**：
- 在光源→屏幕中心的连线上放置预制的光斑精灵
- 简单、可控

**Ghosts（鬼影）**：
模拟镜头内部多次反射：
```python
for i in range(n_ghosts):
    pos = lerp(light_pos, center, i * 0.2)
    alpha = (1 - i/n_ghosts) ** 2 * intensity
```

### 参考资料
- three.js LensFlare
- Godot LensFlare
- NVIDIA GPU Gems - Lens Effects

### 优先级
中 — 视觉提升明显，但需跑性能测试

---

## 代码重构：统一 TaichiRenderer 调用

### 背景
- `render_taichi()` 函数封装了 `TaichiRenderer` 的创建和渲染
- `render_video()` 直接使用 `TaichiRenderer`，代码重复
- 新增参数（如 `anti_alias`, `aa_strength`）需要在两处都添加

### 任务
- 合并 `render_video()` 中的 `TaichiRenderer` 创建逻辑
- 让 `render_video()` 也使用 `render_taichi()` 或统一的封装函数
