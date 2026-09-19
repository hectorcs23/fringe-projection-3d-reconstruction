"""
Multi-frequency phase unwrapping de First Attempt usando la diferencia
envuelta en cada frecuencia. Esto es lo que deberia hacer el algoritmo:
- Captura a 4 frecuencias (N=2, 4, 14, 60 franjas reales)
- En cada frecuencia, calcula delta_phi_wrapped = wrap(phi_obj - phi_ref)
- Multi-freq unwrap sobre las diferencias envueltas
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_fill_holes, binary_erosion

from phase_shifting import (
    n_step_phase, build_mask, multi_frequency_unwrap_robust,
    wrap_to_pi, anchor_background, phase_to_height,
)

FOTOS_DIR = "data/sphere"
OUT_DIR   = "results/sphere_multifreq"
os.makedirs(OUT_DIR, exist_ok=True)

DOWNSCALE = 1  # data/sphere is already stored at 1/4 of the camera resolution

# Conteos de franjas medidos por FFT (no estimados):
FRINGE_COUNTS = [2, 4, 14, 60]

REF_IDS = {
    0: [169, 170, 171, 172],
    1: [173, 174, 175, 176],
    2: [177, 178, 179, 180],
    3: [181, 182, 183, 184],
}
OBJ_IDS = {
    0: [185, 186, 187, 188],
    1: [189, 190, 191, 192],
    2: [193, 194, 195, 196],
    3: [197, 198, 199, 200],
}


def load(idx):
    path = os.path.join(FOTOS_DIR, f"DSC_{idx:04d}.JPG")
    img = Image.open(path).convert("L")
    w, h = img.size
    img = img.resize((w // DOWNSCALE, h // DOWNSCALE), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64)


def save_map(arr, path, title="", cmap="viridis"):
    plt.figure(figsize=(9, 6))
    im = plt.imshow(arr, cmap=cmap)
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close()


def main():
    print(f"Multi-freq unwrap, conteos REALES = {FRINGE_COUNTS}")
    print(f"Ratios: {[FRINGE_COUNTS[i+1]/FRINGE_COUNTS[i] for i in range(3)]}")

    # Calcular fase envuelta de objeto y referencia en cada frecuencia
    obj_w, obj_mod, obj_dc = [], [], []
    ref_w, ref_mod, ref_dc = [], [], []

    for k in range(4):
        print(f"  Procesando freq {k}...")
        ref_imgs = [load(i) for i in REF_IDS[k]]
        obj_imgs = [load(i) for i in OBJ_IDS[k]]
        pw_r, mo_r, dc_r = n_step_phase(ref_imgs)
        pw_o, mo_o, dc_o = n_step_phase(obj_imgs)
        ref_w.append(pw_r); ref_mod.append(mo_r); ref_dc.append(dc_r)
        obj_w.append(pw_o); obj_mod.append(mo_o); obj_dc.append(dc_o)

    # Fase envuelta de la DIFERENCIA en cada frecuencia
    # Esto automaticamente elimina el carrier del proyector y deja solo
    # la fase inducida por la altura del objeto
    diff_wrapped = [wrap_to_pi(obj_w[k] - ref_w[k]) for k in range(4)]

    # Mascara basada en la modulacion de la frecuencia ALTA
    mod_frac = 0.10
    mask = build_mask(obj_mod[-1], obj_dc[-1], mod_frac=mod_frac)
    mask &= build_mask(ref_mod[-1], ref_dc[-1], mod_frac=mod_frac)
    for dc in obj_dc + ref_dc:
        mask &= (dc > 15) & (dc < 250)
    print(f"Mascara: {mask.mean()*100:.1f}% validos")
    save_map(mask.astype(float), f"{OUT_DIR}/mask.png", "Mascara", "gray")

    # Guardar las diferencias envueltas para inspeccion
    for k, dw in enumerate(diff_wrapped):
        save_map(np.where(mask, dw, np.nan), f"{OUT_DIR}/diff_wrapped_f{k}.png",
                 f"delta_phi envuelta freq {k} (N={FRINGE_COUNTS[k]})",
                 "twilight")

    # Multi-freq unwrap robusto (con filtro mediana al orden)
    print("Unwrapping...")
    delta_phi = multi_frequency_unwrap_robust(
        diff_wrapped, FRINGE_COUNTS, mask=mask, median_size=5
    )
    save_map(np.where(mask, delta_phi, np.nan),
             f"{OUT_DIR}/delta_phi_unwrapped.png",
             "delta_phi desenvuelta (multi-freq)", "RdBu_r")

    # Altura
    Z = phase_to_height(delta_phi, K=1.0)
    Z = anchor_background(Z, mask, percentile=5.0)
    Z[~mask] = 0.0

    # Limpieza
    Z_clean = median_filter(Z, size=3)
    mask_clean = binary_fill_holes(mask)
    mask_clean = binary_erosion(mask_clean, iterations=1)

    save_map(np.where(mask_clean, Z_clean, np.nan),
             f"{OUT_DIR}/Z_height.png",
             "Altura Z (rad, multi-freq)", "terrain")

    z_valid = Z_clean[mask_clean]
    print(f"Z (rad):  rango=[{z_valid.min():.2f}, {z_valid.max():.2f}]  "
          f"std={z_valid.std():.3f}  pico={z_valid.max() - np.percentile(z_valid, 5):.2f}")

    # Perfil horizontal
    h, w = Z_clean.shape
    row = h // 2
    plt.figure(figsize=(12, 5))
    z_line = np.where(mask_clean[row], Z_clean[row], np.nan)
    plt.plot(z_line, 'g-', lw=1.5)
    plt.axhline(0, color='k', lw=0.5)
    plt.xlabel("columna (px)"); plt.ylabel("Z (rad)")
    plt.title(f"Perfil horizontal (multi-freq) - y={row}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/profile.png", dpi=95)
    plt.close()

    np.savez(f"{OUT_DIR}/result.npz",
             Z=Z_clean, mask=mask_clean, delta_phi=delta_phi)
    print(f"Resultados en {OUT_DIR}/")


if __name__ == "__main__":
    main()
