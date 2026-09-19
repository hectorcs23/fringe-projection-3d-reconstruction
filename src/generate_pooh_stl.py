"""STL final de Pooh con calibracion lineal estable."""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.ndimage import median_filter, binary_dilation, binary_fill_holes, binary_erosion

OUT_DIR = "results/pooh"

data = np.load(f"{OUT_DIR}/result_mf.npz")
Z_rad, mask, delta_phi = data["Z"], data["mask"], data["delta_phi"]

L_mm = 1450.0
d_mm = 120.0
CARTULINA_ANCHO_MM = 650.0
N_FRANJAS_ALTA = 70
p_mm = CARTULINA_ANCHO_MM / N_FRANJAS_ALTA
K_lineal = L_mm * p_mm / (2.0 * np.pi * d_mm)
print(f"K lineal = {K_lineal:.3f} mm/rad")

dphi_clean = delta_phi.copy()
p_lo, p_hi = np.percentile(dphi_clean[mask], [1, 99])
dphi_clean = np.clip(dphi_clean, p_lo, p_hi)
dphi_clean[~mask] = 0.0
dphi_clean = median_filter(dphi_clean, size=5)

h_mm = K_lineal * dphi_clean
plane_level = np.percentile(h_mm[mask], 5.0)
h_mm = h_mm - plane_level
if np.percentile(h_mm[mask], 95) < 0:
    h_mm = -h_mm
h_mm[~mask] = 0.0

peak = np.percentile(h_mm[mask], 95)
print(f"Pico Pooh (mm): {peak:.1f}")

obj_mask = h_mm > 0.15 * peak
obj_mask = binary_dilation(obj_mask, iterations=8)
if obj_mask.any():
    rows = np.any(obj_mask, axis=1)
    cols = np.any(obj_mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    m = 25
    y0 = max(0, y0 - m); y1 = min(h_mm.shape[0], y1 + m)
    x0 = max(0, x0 - m); x1 = min(h_mm.shape[1], x1 + m)
else:
    y0, y1, x0, x1 = 0, h_mm.shape[0], 0, h_mm.shape[1]

h_crop = h_mm[y0:y1, x0:x1]
mask_crop = mask[y0:y1, x0:x1]

pixel_pitch = CARTULINA_ANCHO_MM / h_mm.shape[1] * 1.1
print(f"pixel_pitch = {pixel_pitch:.3f} mm/px")

ny_c, nx_c = h_crop.shape
x = np.arange(nx_c) * pixel_pitch
y = np.arange(ny_c) * pixel_pitch
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(15, 11))

ax1 = fig.add_subplot(221)
im = ax1.imshow(np.where(mask_crop, h_crop, np.nan),
                cmap="terrain",
                extent=[0, nx_c*pixel_pitch, ny_c*pixel_pitch, 0])
ax1.set_title(f"Altura Z (mm) - Pooh")
ax1.set_xlabel("x (mm)"); ax1.set_ylabel("y (mm)")
plt.colorbar(im, ax=ax1, label="Z (mm)")

ax2 = fig.add_subplot(222, projection='3d')
step = 3
X_d = X[::step, ::step]; Y_d = Y[::step, ::step]; Z_d = h_crop[::step, ::step]
m_d = mask_crop[::step, ::step]
Z_d_masked = np.where(m_d, Z_d, np.nan)
ax2.plot_surface(X_d, Y_d, Z_d_masked, cmap="terrain", edgecolor="none",
                 rstride=1, cstride=1, antialiased=True)
ax2.set_title("Vista 3D")
ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)"); ax2.set_zlabel("Z (mm)")
ax2.view_init(elev=25, azim=-70)

ax3 = fig.add_subplot(223)
peak_row = np.argmax(np.where(mask_crop, h_crop, 0).max(axis=1))
cols_v = np.where(mask_crop[peak_row])[0]
if len(cols_v) > 0:
    ax3.plot(x[cols_v], h_crop[peak_row, cols_v], 'b-', lw=2)
ax3.set_title(f"Perfil horizontal (fila pico)")
ax3.set_xlabel("x (mm)"); ax3.set_ylabel("Z (mm)")
ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(224)
peak_col = np.argmax(np.where(mask_crop, h_crop, 0).max(axis=0))
rows_v = np.where(mask_crop[:, peak_col])[0]
if len(rows_v) > 0:
    ax4.plot(y[rows_v], h_crop[rows_v, peak_col], 'g-', lw=2)
ax4.set_title(f"Perfil vertical (col pico)")
ax4.set_xlabel("y (mm)"); ax4.set_ylabel("Z (mm)")
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pooh_3d_final.png", dpi=100)
plt.close()

z_valid = h_crop[mask_crop]
print(f"\nAltura Pooh (mm): rango=[{z_valid.min():.1f}, {z_valid.max():.1f}]  pico={z_valid.max() - np.percentile(z_valid, 5):.1f}")

import sys
sys.path.insert(0, ".")
from SuperficieSTL import heightmap_to_mesh, smooth_mesh, export_stl
h_for_stl = h_crop.copy()
h_for_stl[~mask_crop] = 0.0
mesh = heightmap_to_mesh(X, Y, h_for_stl, mask_crop)
mesh = smooth_mesh(mesh, iterations=8)
export_stl(mesh, f"{OUT_DIR}/pooh.stl")
print(f"STL: {OUT_DIR}/pooh.stl")
