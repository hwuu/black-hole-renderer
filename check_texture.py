#!/usr/bin/env python3
import numpy as np
from PIL import Image
import sys
import time
sys.path.insert(0, '.')

print("Starting...")
t0 = time.time()

import render

print(f"Import done: {time.time() - t0:.2f}s")
t1 = time.time()

print("Generating texture in polar coordinates...")
r_inner = 2.0
r_outer = 12.0
tex = render.generate_disk_texture(n_phi=1024, n_r=512, seed=42, r_inner=r_inner, r_outer=r_outer, enable_rt=True)

print(f"Texture generated: {time.time() - t1:.2f}s")

n_r, n_phi, _ = tex.shape
print(f"n_r={n_r}, n_phi={n_phi}")

rgb = tex[..., :3]
Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save('output/disk_texture_polar.png')
print(f"Saved to output/disk_texture_polar.png (polar, {n_phi}x{n_r})")

size = 1024
print(f"Creating output image {size}x{size}...")
lin = np.linspace(-r_outer, r_outer, size)
xv, yv = np.meshgrid(lin, lin[::-1])
r = np.sqrt(xv**2 + yv**2)
phi = np.mod(np.arctan2(yv, xv) + 2*np.pi, 2*np.pi)

mask = (r >= r_inner) & (r <= r_outer)

phi_idx = phi / (2 * np.pi) * n_phi
r_idx = (r - r_inner) / (r_outer - r_inner) * (n_r - 1)

phi0 = np.floor(phi_idx).astype(int) % n_phi
r0 = np.clip(np.floor(r_idx).astype(int), 0, n_r - 1)
phi1 = (phi0 + 1) % n_phi
r1 = np.clip(r0 + 1, 0, n_r - 1)

f_phi = (phi_idx - np.floor(phi_idx)).astype(np.float32)
f_r = (r_idx - np.floor(r_idx)).astype(np.float32)

canvas = np.zeros((size, size, 3), dtype=np.float32)
for c in range(3):
    c00 = tex[r0, phi0, c]
    c10 = tex[r0, phi1, c]
    c01 = tex[r1, phi0, c]
    c11 = tex[r1, phi1, c]
    val = c00*(1-f_phi)*(1-f_r) + c10*f_phi*(1-f_r) + c01*(1-f_phi)*f_r + c11*f_phi*f_r
    canvas[..., c] = np.where(mask, val, 0.0)

Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8)).save('output/disk_texture_with_rt.png')
print(f"Saved to output/disk_texture_with_rt.png, total: {time.time() - t0:.2f}s")