#!/usr/bin/env bash
# V2 视觉验收脚本：生成 V1 技术对照与 V2 acceptance（volume 主路径）。
#
# D2（2026-06-14）：v2 主验收参数升级到 v2.2 推荐值：
#   --ar1 3 --ar2 50 --pov "30 0 10"
# v2.1 design 实算确认 ar2=15 温度跨度只有约 1.5 倍、ar2=50 可达 4.32 倍；
# v2.2 design §2.4 "内圈紫蓝 → 中段橙黄 → 外圈暗红" 的色阶承诺需要大盘才能拿到。
# 小盘 (ar1=2, ar2=15) 与 V1 默认行为保留为单独的对照图，便于横向比较。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p output

PYTHON="${PYTHON:-python}"
if command -v conda >/dev/null 2>&1; then
  PYTHON="conda run -n black-hole python"
fi

echo "[acceptance] 生成暗色 skybox（若不存在）..."
if [[ ! -f output/black_sky.png ]]; then
  $PYTHON - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

path = Path("output/black_sky.png")
path.parent.mkdir(parents=True, exist_ok=True)
# 暗蓝黑背景，避免星空抢盘面
rgb = np.zeros((1024, 2048, 3), dtype=np.uint8)
rgb[..., 0] = 5
rgb[..., 1] = 5
rgb[..., 2] = 18
Image.fromarray(rgb, mode="RGB").save(path)
print(f"Wrote {path}")
PY
fi

# --- 主验收参数（v2.2 推荐：r_out=50 让温度跨度达 4.32 倍） ---
# 相机距离 ~100 r_s、仰角 ~18.6° 保持与 disk_tilt=20° 接近的平视角度，
# FOV=90° 下盘外缘 (r_out=50) 投影约占屏幕高度 60%~70%，留出黑洞透镜余地。
ACCEPT_POV="95 0 32"
ACCEPT_FOV=90
ACCEPT_AR1=3
ACCEPT_AR2=50
ACCEPT_TILT=20
SKY="output/black_sky.png"

ACCEPT_COMMON=(
  --texture "$SKY"
  --pov $ACCEPT_POV
  --fov "$ACCEPT_FOV"
  --ar1 "$ACCEPT_AR1"
  --ar2 "$ACCEPT_AR2"
  --disk_tilt "$ACCEPT_TILT"
  -r hd
  --device gpu
)

# --- 小盘对照参数（v1 默认行为 + V1 兼容范围，用于横向比较） ---
COMPAT_POV="24 0 8"
COMPAT_FOV=90
COMPAT_AR1=2
COMPAT_AR2=15
COMPAT_TILT=20

COMPAT_COMMON=(
  --texture "$SKY"
  --pov $COMPAT_POV
  --fov "$COMPAT_FOV"
  --ar1 "$COMPAT_AR1"
  --ar2 "$COMPAT_AR2"
  --disk_tilt "$COMPAT_TILT"
  -r hd
  --device gpu
)

echo "[acceptance] V1 classic 技术对照（小盘 ar2=15，非最终美术标准）..."
$PYTHON render.py \
  "${COMPAT_COMMON[@]}" \
  -o output/v1_classic_darksky.png

echo "[acceptance] V2 Step 0 基线（volume 主路径，无 bloom，主验收参数）..."
$PYTHON render.py \
  --disk_model v2 \
  "${ACCEPT_COMMON[@]}" \
  --v2_visual_preset interstellar \
  --v2_disable_visual_atlas \
  --v2_bloom_intensity 0 \
  --v2_print_stats \
  -o output/v2_step0_baseline.png

echo "[acceptance] V2 acceptance — no bloom（volume 主路径，主验收参数）..."
$PYTHON render.py \
  --disk_model v2 \
  "${ACCEPT_COMMON[@]}" \
  --v2_visual_preset interstellar \
  --v2_disable_visual_atlas \
  --v2_bloom_intensity 0 \
  --v2_print_stats \
  -o output/v2_acceptance_no_bloom.png

echo "[acceptance] V2 acceptance — bloom（volume 主路径，主验收参数）..."
$PYTHON render.py \
  --disk_model v2 \
  "${ACCEPT_COMMON[@]}" \
  --v2_visual_preset interstellar \
  --v2_disable_visual_atlas \
  --v2_print_stats \
  -o output/v2_acceptance_bloom.png

echo "[acceptance] V2 atlas thin-layer 对照（主验收参数，非主验收）..."
$PYTHON render.py \
  --disk_model v2 \
  "${ACCEPT_COMMON[@]}" \
  --v2_visual_preset interstellar \
  --v2_print_stats \
  -o output/v2_acceptance_atlas.png

echo "[acceptance] V2 小盘对照（v1 兼容范围 ar1=2 ar2=15，volume 主路径）..."
$PYTHON render.py \
  --disk_model v2 \
  "${COMPAT_COMMON[@]}" \
  --v2_visual_preset interstellar \
  --v2_disable_visual_atlas \
  --v2_bloom_intensity 0 \
  --v2_print_stats \
  -o output/v2_compat_small_disk.png

echo "[acceptance] 完成。请人工对照参考图判断构图/纹理/色调/光效。"
echo ""
echo "  主验收（v2.2 推荐参数 r_out=50）："
echo "  - output/v2_step0_baseline.png"
echo "  - output/v2_acceptance_no_bloom.png"
echo "  - output/v2_acceptance_bloom.png"
echo ""
echo "  对照（非主验收）："
echo "  - output/v1_classic_darksky.png         # V1 技术对照（小盘）"
echo "  - output/v2_acceptance_atlas.png        # V2 atlas thin-layer（主验收参数）"
echo "  - output/v2_compat_small_disk.png       # V2 volume 路径在小盘下的表现"
