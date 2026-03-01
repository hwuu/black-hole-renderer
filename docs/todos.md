# TODO List

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
