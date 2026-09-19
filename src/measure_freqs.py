"""Medir la frecuencia real de cada set de referencia usando FFT."""
from pathlib import Path
import re

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
FOTOS_DIR = SCRIPT_DIR.parent / "data" / "sphere"
DOWNSCALE = 1

# IDs originales. Si esas fotos no existen, el script detecta las fotos
# disponibles en FOTOS_DIR y usa las primeras 16, agrupadas en bloques de 4.
REF_IDS = {
    0: [169, 170, 171, 172],
    1: [173, 174, 175, 176],
    2: [177, 178, 179, 180],
    3: [181, 182, 183, 184],
}

DSC_RE = re.compile(r"DSC_(\d{4})", re.IGNORECASE)


def path_for(idx):
    path = FOTOS_DIR / f"DSC_{idx:04d}.JPG"
    if path.exists():
        return path

    matches = list(FOTOS_DIR.glob(f"DSC_{idx:04d}.*"))
    if matches:
        return matches[0]
    return path


def available_ids():
    if not FOTOS_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de fotos: {FOTOS_DIR}")

    ids = []
    for path in FOTOS_DIR.glob("DSC_*.*"):
        m = DSC_RE.search(path.stem)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def resolve_ref_ids():
    missing = [idx for group in REF_IDS.values() for idx in group if not path_for(idx).exists()]
    if not missing:
        return REF_IDS

    ids = available_ids()
    if len(ids) < 16:
        raise FileNotFoundError(
            f"No se encontraron las fotos configuradas {missing} y solo hay "
            f"{len(ids)} fotos DSC disponibles en {FOTOS_DIR}."
        )

    selected = ids[:16]
    print(
        "Aviso: las fotos configuradas no existen; usando las primeras 16 "
        f"fotos disponibles: DSC_{selected[0]:04d} a DSC_{selected[-1]:04d}."
    )
    return {k: selected[4 * k : 4 * k + 4] for k in range(4)}


def load(idx):
    path = path_for(idx)
    img = Image.open(path).convert("L")
    w, h = img.size
    img = img.resize((w // DOWNSCALE, h // DOWNSCALE), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64)


def measure_fringe_count(imgs):
    """Cuenta cuantas franjas hay a lo largo del eje X (franjas verticales)."""
    img = imgs[0]
    h, w = img.shape
    strip = img[h // 2 - 50 : h // 2 + 50].mean(axis=0)
    strip = strip - strip.mean()
    strip = strip[w // 4 : 3 * w // 4]

    spectrum = np.abs(np.fft.rfft(strip))
    spectrum[0] = 0
    peak_bin = np.argmax(spectrum)

    analysis_width = len(strip)
    total_width = w
    cycles_in_total_image = peak_bin * (total_width / analysis_width)
    return cycles_in_total_image


print("Conteo real de franjas (frecuencia dominante por FFT):")
for k, ids in resolve_ref_ids().items():
    imgs = [load(i) for i in ids]
    count = measure_fringe_count(imgs)
    print(f"  freq {k} (fotos {ids[0]}-{ids[-1]}): ~{count:.1f} franjas")
