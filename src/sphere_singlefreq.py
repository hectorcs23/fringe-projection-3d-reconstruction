"""
Single-frequency usando la freq ALTA (60 franjas) para First Attempt.
Con foco nítido, esto debería dar la mejor reconstrucción.
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_fill_holes, binary_erosion
from mpl_toolkits.mplot3d import Axes3D  # noqa

from phase_shifting import (
    n_step_phase, wrap_to_pi, unwrap_2d,
    anchor_background, phase_to_height,
)

FOTOS_DIR = "data/sphere"
OUT_DIR   = "results/sphere_singlefreq"
os.makedirs(OUT_DIR, exist_ok=True)

DOWNSCALE = 1  # data/sphere is already stored at 1/4 of the camera resolution

# Probar distintas frecuencias
FREQ_SETS = {
    "f1_low":  {"ref": [173, 174, 175, 176], "obj": [189, 190, 191, 192]},
    "f2_mid":  {"ref": [177, 178, 179, 180], "obj": [193, 194, 195, 196]},
    "f3_high": {"ref": [181, 182, 183, 184], "obj": [197, 198, 199, 200]},
}


def load(idx):
    path = os.path.join(FOTOS_DIR, f"DSC_{idx:04d}.JPG")
    img = Image.open(path).convert("L")
    w, h = img.size
    img = img.resize((w // DOWNSCALE, h // DOWNSCALE), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64)


def process_single(ref_ids, obj_ids, mod_frac=0.15):
    ref_imgs = [load(i) for i in ref_ids]
    obj_imgs = [load(i) for i in obj_ids]

    phi_ref_w, mod_ref, dc_ref = n_step_phase(ref_imgs)
    phi_obj_w, mod_obj, dc_obj = n_step_phase(obj_imgs)

    # Mascara basada en modulacion
    mask = mod_ref > mod_frac * np.percentile(mod_ref, 99)
    mask &= mod_obj > mod_frac * np.percentile(mod_obj, 99)
    mask &= (dc_obj > 15) & (dc_obj < 250)
    mask &= (dc_ref > 15) & (dc_ref < 250)

    # Unwrappar la diferencia envuelta directamente
    delta_phi_w = wrap_to_pi(phi_obj_w - phi_ref_w)
    delta_phi = unwrap_2d(delta_phi_w, mask)

    return delta_phi, mask, mod_obj, phi_obj_w, phi_ref_w


def main():
    results = {}
    for name, ids in FREQ_SETS.items():
        print(f"{name}...")
        dphi, mask, mod, _, _ = process_single(ids["ref"], ids["obj"])
        Z = phase_to_height(dphi, K=1.0)
        Z = anchor_background(Z, mask, percentile=5.0)
        z_valid = Z[mask]
        print(f"  validos={mask.mean()*100:.1f}%  "
              f"Z range=[{z_valid.min():.2f}, {z_valid.max():.2f}]  "
              f"std={z_valid.std():.2f}")
        results[name] = (dphi, mask, Z)

    # Figura comparativa
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    for i, (name, (dphi, mask, Z)) in enumerate(results.items()):
        Zv = np.where(mask, Z, np.nan)
        im0 = axes[i, 0].imshow(dphi, cmap="viridis")
        axes[i, 0].set_title(f"{name}: delta_phi (rad)")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.04)

        im1 = axes[i, 1].imshow(Zv, cmap="terrain")
        axes[i, 1].set_title(f"{name}: altura Z (rad)")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.04)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/single_freq_comparison.png", dpi=85)
    plt.close()

    # Perfiles
    plt.figure(figsize=(12, 5))
    for name, (dphi, mask, Z) in results.items():
        h, w = Z.shape
        row = h // 2
        zline = np.where(mask[row], Z[row], np.nan)
        plt.plot(zline, label=name, lw=1.5)
    plt.axhline(0, color='k', lw=0.5)
    plt.xlabel("columna (px)"); plt.ylabel("Z (rad)")
    plt.title("Perfil horizontal por el centro")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/single_freq_profiles.png", dpi=95)
    plt.close()

    # Guardar el mejor (f3_high) para seguir con STL
    dphi, mask, Z = results["f3_high"]
    np.savez(f"{OUT_DIR}/best_single.npz", Z=Z, mask=mask, delta_phi=dphi)
    print(f"\nResultados: {OUT_DIR}/single_freq_*.png")


if __name__ == "__main__":
    main()
