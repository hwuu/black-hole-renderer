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

print("Generating texture...")
tex = render.generate_disk_texture(2048, 1024, seed=42)

print(f"Texture generated: {time.time() - t1:.2f}s")
t2 = time.time()

tex_h, tex_w, _ = tex.shape
print(f"tex_h={tex_h}, tex_w={tex_w}")

r_inner = 2.0
r_outer = 12.0

size = 2048
print(f"Creating meshgrid {size}x{size}...")
lin = np.linspace(-r_outer, r_outer, size)
xv, yv = np.meshgrid(lin, lin[::-1])
r = np.sqrt(xv**2 + yv**2)

print(f"Meshgrid done: {time.time() - t2:.2f}s")
t3 = time.time()

mask = (r >= r_inner) & (r <= r_outer)
phi = np.mod(np.arctan2(yv, xv) + 2*np.pi, 2*np.pi)
u = phi / (2*np.pi) * tex_w
vv = (r - r_inner) / (r_outer - r_inner) * tex_h

print(f"UV computed: {time.time() - t3:.2f}s")
t4 = time.time()

nu0 = np.floor(u).astype(int) % tex_w
nv0 = np.clip(np.floor(vv).astype(int), 0, tex_h-1)
nu1 = (nu0 + 1) % tex_w
nv1 = np.clip(nv0 + 1, 0, tex_h-1)

fu = (u - np.floor(u)).astype(np.float32)
fv = (vv - np.floor(vv)).astype(np.float32)

print(f"Index done: {time.time() - t4:.2f}s")
t5 = time.time()

print("Testing single channel indexing...")
test = tex[nv0, nu0, 0]
print(f"Single channel test done: {time.time() - t5:.2f}s")

canvas = np.zeros((size, size, 3), dtype=np.float32)
for c in range(3):
    c00 = tex[nv0, nu0, c]
    c10 = tex[nv0, nu1, c]
    c01 = tex[nv1, nu0, c]
    c11 = tex[nv1, nu1, c]
    val = c00*(1-fu)*(1-fv) + c10*fu*(1-fv) + c01*(1-fu)*fv + c11*fu*fv
    canvas[..., c] = np.where(mask, val, 0.0)

print(f"Bilinear done: {time.time() - t5:.2f}s")
t6 = time.time()

Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8)).save('output/disk_topdown_ar2_12.png')
print(f"Saved to output/disk_topdown_ar2_12.png, total: {time.time() - t0:.2f}s")