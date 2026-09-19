"""
Pipeline para Third Attempt (Pooh, DSC_0396-0427).

Estructura asumida:
  Objeto (Pooh):
    f0 (baja):  396, 397, 398, 399
    f1:         400, 401, 402, 403
    f2:         404, 405, 406, 407
    f3 (alta):  408, 409, 410, 411
  Referencia (cartulina):
    f0: 412, 413, 414, 415
    f1: 416, 417, 418, 419
    f2: 420, 421, 422, 423
    f3: 424, 425, 426, 427
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_fill_holes, binary_erosion

from phase_shifting import (
    n_step_phase, build_mask, multi_frequency_unwrap_robust,
    wrap_to_pi, anchor_background, phase_to_height, unwrap_2d,
)

FOTOS_DIR = "data/pooh_front"
OUT_DIR   = "results/pooh"
os.makedirs(OUT_DIR, exist_ok=True)

DOWNSCALE = 2  # las fotos ya estaban comprimidas a 1/4, otro 1/2 = 1/8 final

OBJ_IDS = {
    0: [396, 397, 398, 399],
    1: [400, 401, 402, 403],
    2: [404, 405, 406, 407],
    3: [408, 409, 410, 411],
}
REF_IDS = {
    0: [412, 413, 414, 415],
    1: [416, 417, 418, 419],
    2: [420, 421, 422, 423],
    3: [424, 425, 426, 427],
}


def load(idx):
    path = os.path.join(FOTOS_DIR, f"DSC_{idx:04d}.JPG")
    img = Image.open(path).convert("L")
    if DOWNSCALE > 1:
        w, h = img.size
        img = img.resize((w // DOWNSCALE, h // DOWNSCALE), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64)


def save_map(arr, path, title="", cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(10, 7))
    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=85)
    plt.close()


def measure_freq_via_fft(imgs, axis="vertical_fringes"):
    """Medir cuantas franjas verticales hay (varia en X)."""
    img = imgs[0]
    h, w = img.shape
    strip = img[h//2 - 50 : h//2 + 50].mean(axis=0)
    strip = strip - strip.mean()
    strip = strip[w//4 : 3*w//4]
    spectrum = np.abs(np.fft.rfft(strip))
    spectrum[0] = 0
    peak_bin = np.argmax(spectrum)
    return peak_bin * (w / len(strip))


def main():
    # Medir frecuencias reales con FFT en la referencia
    print("Midiendo frecuencias por FFT...")
    fringe_counts = []
    for k in REF_IDS:
        imgs = [load(i) for i in REF_IDS[k]]
        f = measure_freq_via_fft(imgs)
        fringe_counts.append(round(f))
        print(f"  freq {k}: ~{f:.1f} franjas")
    # Normalizar para asegurar que sean estrictamente crecientes
    fringe_counts = sorted(set(fringe_counts))
    while len(fringe_counts) < 4:
        fringe_counts.append(fringe_counts[-1] + 1)
    print(f"  -> usando: {fringe_counts}")

    # Procesar
    obj_w, obj_mod, obj_dc = [], [], []
    ref_w, ref_mod, ref_dc = [], [], []

    for k in range(4):
        print(f"Procesando freq {k}...")
        ref_imgs = [load(i) for i in REF_IDS[k]]
        obj_imgs = [load(i) for i in OBJ_IDS[k]]
        pw_r, mo_r, dc_r = n_step_phase(ref_imgs)
        pw_o, mo_o, dc_o = n_step_phase(obj_imgs)
        ref_w.append(pw_r); ref_mod.append(mo_r); ref_dc.append(dc_r)
        obj_w.append(pw_o); obj_mod.append(mo_o); obj_dc.append(dc_o)
        save_map(mo_o, f"{OUT_DIR}/obj_f{k}_mod.png",
                 f"obj f{k} mod", "magma")

    # Diferencia envuelta en cada freq
    diff_wrapped = [wrap_to_pi(obj_w[k] - ref_w[k]) for k in range(4)]
    for k, dw in enumerate(diff_wrapped):
        save_map(dw, f"{OUT_DIR}/diff_wrap_f{k}.png",
                 f"diff envuelta f{k} (N={fringe_counts[k]})",
                 "twilight")

    # Mascara: menos estricta que con la esfera porque Pooh tiene menos contraste
    mod_frac = 0.08
    mask = build_mask(obj_mod[-1], obj_dc[-1], mod_frac=mod_frac)
    mask &= build_mask(ref_mod[-1], ref_dc[-1], mod_frac=mod_frac)
    for dc in obj_dc + ref_dc:
        mask &= (dc > 10) & (dc < 250)
    print(f"Mascara: {mask.mean()*100:.1f}% validos")
    save_map(mask.astype(float), f"{OUT_DIR}/mask.png", "Mascara", "gray")

    # Multi-freq unwrap con filtro robusto
    print("Multi-freq unwrap...")
    delta_phi = multi_frequency_unwrap_robust(
        diff_wrapped, fringe_counts, mask=mask, median_size=5
    )

    # Recortar outliers extremos para visualizar
    p_lo, p_hi = np.percentile(delta_phi[mask], [1, 99])
    save_map(np.where(mask, delta_phi, np.nan),
             f"{OUT_DIR}/delta_phi_unwrapped.png",
             "delta_phi desenvuelta", "RdBu_r",
             vmin=p_lo, vmax=p_hi)

    # Altura
    Z = phase_to_height(delta_phi, K=1.0)
    Z = anchor_background(Z, mask, percentile=5.0)
    Z[~mask] = 0.0

    # Limpieza
    Z_clean = median_filter(Z, size=3)
    mask_clean = binary_fill_holes(mask)
    mask_clean = binary_erosion(mask_clean, iterations=1)

    p_lo, p_hi = np.percentile(Z_clean[mask_clean], [1, 99])
    save_map(np.where(mask_clean, Z_clean, np.nan),
             f"{OUT_DIR}/Z_height.png",
             f"Altura Z (rad, multi-freq)\n"
             , vmin=p_lo, vmax=p_hi)

    # Tambien probar single-frequency para comparar
    print("Single-freq con freq alta...")
    delta_phi_sf = unwrap_2d(diff_wrapped[-1], mask)
    Z_sf = anchor_background(delta_phi_sf, mask, percentile=5.0)
    Z_sf[~mask] = 0.0
    Z_sf_clean = median_filter(Z_sf, size=3)

    p_lo_sf, p_hi_sf = np.percentile(Z_sf_clean[mask_clean], [1, 99])
    save_map(np.where(mask_clean, Z_sf_clean, np.nan),
             f"{OUT_DIR}/Z_height_singlefreq.png",
             f"Altura Z (single-freq)\n"
             , vmin=p_lo_sf, vmax=p_hi_sf)

    z_valid = Z_clean[mask_clean]
    z_valid_sf = Z_sf_clean[mask_clean]
    print(f"\nMulti-freq Z (rad): rango=[{z_valid.min():.2f}, {z_valid.max():.2f}]")
    print(f"Single-freq Z (rad): rango=[{z_valid_sf.min():.2f}, {z_valid_sf.max():.2f}]")

    # Guardar
    np.savez(f"{OUT_DIR}/result_mf.npz",
             Z=Z_clean, mask=mask_clean, delta_phi=delta_phi)
    np.savez(f"{OUT_DIR}/result_sf.npz",
             Z=Z_sf_clean, mask=mask_clean, delta_phi=delta_phi_sf)
    print(f"\nResultados en {OUT_DIR}/")


if __name__ == "__main__":
    main()
