#!/usr/bin/env bash
# V2 视觉验收脚本：生成固定相机下的 V1 技术对照与 V2 acceptance（no-bloom / bloom）。
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

ACCEPT_POV="24 0 8"
ACCEPT_FOV=90
ACCEPT_AR1=2
ACCEPT_AR2=15
ACCEPT_TILT=20
SKY="output/black_sky.png"
COMMON=(
  --texture "$SKY"
  --pov $ACCEPT_POV
  --fov "$ACCEPT_FOV"
  --ar1 "$ACCEPT_AR1"
  --ar2 "$ACCEPT_AR2"
  --disk_tilt "$ACCEPT_TILT"
  -r hd
  --device gpu
)

echo "[acceptance] V1 classic 技术对照（非最终美术标准）..."
$PYTHON render.py \
  "${COMMON[@]}" \
  -o output/v1_classic_darksky.png

echo "[acceptance] V2 Step 0 基线（visual atlas，无 bloom）..."
$PYTHON render.py \
  --disk_model v2 \
  "${COMMON[@]}" \
  --v2_visual_preset interstellar \
  --v2_bloom_intensity 0 \
  --v2_print_stats \
  -o output/v2_step0_baseline.png

echo "[acceptance] V2 acceptance — no bloom..."
$PYTHON render.py \
  --disk_model v2 \
  "${COMMON[@]}" \
  --v2_visual_preset interstellar \
  --v2_bloom_intensity 0 \
  --v2_print_stats \
  -o output/v2_acceptance_no_bloom.png

echo "[acceptance] V2 acceptance — bloom..."
$PYTHON render.py \
  --disk_model v2 \
  "${COMMON[@]}" \
  --v2_visual_preset interstellar \
  --v2_print_stats \
  -o output/v2_acceptance_bloom.png

echo "[acceptance] 完成。请人工对照参考图判断构图/纹理/色调/光效。"
echo "  - output/v1_classic_darksky.png"
echo "  - output/v2_step0_baseline.png"
echo "  - output/v2_acceptance_no_bloom.png"
echo "  - output/v2_acceptance_bloom.png"
