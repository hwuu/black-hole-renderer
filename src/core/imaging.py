import os
from PIL import Image
import numpy as np


def save_image(image: np.ndarray, path: str) -> None:
    """保存图像为 PNG 文件。

    Args:
        image: float LDR `[0, 1]` 或 uint8 `[0, 255]` RGB，形状 `(H, W, 3)`。
        path: 输出 PNG 路径。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if image.dtype == np.uint8:
        img_uint8 = image
    else:
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8, "RGB").save(path)
    print(f"Saved: {path}")
