"""Comparar multi-freq robusto vs single-freq."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

mf = np.load("results/sphere_multifreq/result.npz")
sf = np.load("results/sphere_singlefreq/best_single.npz")

Z_mf, mask_mf = mf["Z"], mf["mask"]
Z_sf, mask_sf = sf["Z"], sf["mask"]

# Robusto contra outliers: aplicar un clip al percentil 99.5
p_low, p_hi = np.percentile(Z_mf[mask_mf], [0.5, 99.5])
Z_mf_clip = np.clip(Z_mf, p_low, p_hi)

# Estadisticas
print("MULTI-FRECUENCIA:")
print(f"  Z crudo range: [{Z_mf[mask_mf].min():.2f}, {Z_mf[mask_mf].max():.2f}]")
print(f"  p0.5-p99.5:    [{p_low:.3f}, {p_hi:.3f}]")
print(f"  Outliers > p99.5: {(Z_mf[mask_mf] > p_hi).sum()} pixeles "
      f"({(Z_mf[mask_mf] > p_hi).mean()*100:.3f}%)")
print(f"  Outliers < p0.5: {(Z_mf[mask_mf] < p_low).sum()} pixeles")

print("\nSINGLE-FRECUENCIA (f=60):")
print(f"  Z range: [{Z_sf[mask_sf].min():.2f}, {Z_sf[mask_sf].max():.2f}]")

# Figura comparativa
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

Z_mf_vis = np.where(mask_mf, Z_mf_clip, np.nan)
Z_sf_vis = np.where(mask_sf, Z_sf, np.nan)

im = axes[0, 0].imshow(Z_mf_vis, cmap="terrain")
axes[0, 0].set_title(f"MULTI-FRECUENCIA\n"
                     f"N={[2,4,14,60]} franjas")
plt.colorbar(im, ax=axes[0, 0], fraction=0.04, label="Z (rad)")

im = axes[0, 1].imshow(Z_sf_vis, cmap="terrain")
axes[0, 1].set_title(f"SINGLE-FRECUENCIA (f=60)")
plt.colorbar(im, ax=axes[0, 1], fraction=0.04, label="Z (rad)")

# Perfiles horizontales
h, w = Z_mf.shape
row = h // 2
ax = axes[1, 0]
z_mf_line = np.where(mask_mf[row], Z_mf_clip[row], np.nan)
z_sf_line = np.where(mask_sf[row], Z_sf[row], np.nan)
ax.plot(z_mf_line, 'b-', lw=1.5, label="Multi-freq")
ax.plot(z_sf_line, 'r-', lw=1.5, alpha=0.7, label="Single-freq")
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel("columna (px)"); ax.set_ylabel("Z (rad)")
ax.set_title(f"Perfil horizontal y={row}")
ax.legend(); ax.grid(alpha=0.3)

# Diferencia entre metodos (solo donde ambos son validos)
both_valid = mask_mf & mask_sf
diff = Z_mf_clip - Z_sf
diff[~both_valid] = np.nan
ax = axes[1, 1]
im = ax.imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
ax.set_title("Diferencia Multi - Single (rad)")
plt.colorbar(im, ax=ax, fraction=0.04)

plt.tight_layout()
plt.savefig("results/sphere_multifreq/comparison_mf_vs_sf.png", dpi=95)
plt.close()

# Estadisticas de acuerdo entre metodos en la esfera
if both_valid.any():
    d = np.abs(Z_mf_clip - Z_sf)[both_valid]
    print(f"\nACUERDO (diferencia absoluta):")
    print(f"  mediana = {np.median(d):.4f} rad")
    print(f"  p95     = {np.percentile(d, 95):.4f} rad")

print("\nGuardado: results/sphere_multifreq/comparison_mf_vs_sf.png")
