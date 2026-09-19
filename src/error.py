"""
error.py

Analisis de error para reconstruccion multifrecuencia / single-frequency.

Este archivo asume que primero corriste tu pipeline principal y que existen:

    out_pooh/result_mf.npz
    out_pooh/result_sf.npz

Cada .npz debe contener:
    Z      -> mapa reconstruido en fase/radianes o altura relativa
    mask   -> mascara valida

El script:
    1. Carga Z multi-freq y single-freq.
    2. Te pide hacer clic en los puntos medidos.
    3. Extrae la mediana en una ROI alrededor de cada punto.
    4. Calibra fase -> cm.
    5. Calcula errores contra tus mediciones reales.
    6. Guarda CSV, graficas y mapas calibrados.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACION
# ============================================================

OUT_DIR = "results/pooh"

MF_FILE = os.path.join(OUT_DIR, "result_mf.npz")
SF_FILE = os.path.join(OUT_DIR, "result_sf.npz")

# Geometria del sistema, en cm
B_CM = 15.0     # separacion camara-proyector
L_CM = 142.0    # distancia proyector/camara a pantalla

# Radio de la ROI en pixeles.
# Si el mapa esta ruidoso, prueba 10, 12 o 15.
ROI_HALF = 8

# Archivo donde se guardan los puntos seleccionados para no tener que
# hacer clic cada vez.
POINTS_JSON = os.path.join(OUT_DIR, "roi_points.json")

# Mediciones reales en cm.
# Se interpretan como altura/profundidad desde la pantalla de referencia.
Z_REAL_CM = {
    "Pie Derecho": 23.0,
    "Pie Izquierdo": 23.5,
    "Mano Derecha": 14.0,
    "Mano Izquierda": 14.5,
    "Oreja Derecha": 5.0,
    "Oreja Izquierda": 4.8,
    "Ojo Derecho": 13.3,
    "Ojo Izquierdo": 13.3,
    "Nariz": 17.8,
    "Panza": 16.0,
    "Camisa": 14.5,
}


# ============================================================
# FUNCIONES
# ============================================================

def check_input_files():
    """Verifica que existan los archivos .npz del pipeline principal."""
    missing = []
    if not os.path.exists(MF_FILE):
        missing.append(MF_FILE)
    if not os.path.exists(SF_FILE):
        missing.append(SF_FILE)

    if missing:
        print("\nERROR: No se encontraron estos archivos:")
        for f in missing:
            print(f"  - {f}")

        print("\nPrimero corre tu pipeline principal, el que genera:")
        print("  out_pooh/result_mf.npz")
        print("  out_pooh/result_sf.npz")
        print("\nEjemplo:")
        print("  python tu_pipeline_pooh.py")
        raise FileNotFoundError("Faltan archivos .npz de entrada.")


def load_results():
    """Carga mapas multi-freq, single-freq y mascara."""
    check_input_files()

    mf = np.load(MF_FILE)
    sf = np.load(SF_FILE)

    if "Z" not in mf or "mask" not in mf:
        raise KeyError(f"{MF_FILE} debe contener las llaves 'Z' y 'mask'.")

    if "Z" not in sf:
        raise KeyError(f"{SF_FILE} debe contener la llave 'Z'.")

    z_phase_mf = mf["Z"].astype(float).copy()
    z_phase_sf = sf["Z"].astype(float).copy()
    mask_clean = mf["mask"].astype(bool)

    if z_phase_mf.shape != z_phase_sf.shape:
        raise ValueError("Z multi-freq y Z single-freq no tienen el mismo tamaño.")

    if z_phase_mf.shape != mask_clean.shape:
        raise ValueError("Z multi-freq y mask no tienen el mismo tamaño.")

    return z_phase_mf, z_phase_sf, mask_clean


def save_map(arr, path, title="", cmap="viridis", vmin=None, vmax=None):
    """Guarda un mapa 2D como imagen."""
    plt.figure(figsize=(10, 7))
    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def pick_rois(img, labels):
    """
    Permite seleccionar puntos con clic.
    Haz clic en cada punto en el orden indicado.
    Regresa diccionario:
        {label: (x, y)}
    """
    finite = np.isfinite(img)
    if finite.any():
        vmin, vmax = np.nanpercentile(img[finite], [2, 98])
    else:
        vmin, vmax = None, None

    plt.figure(figsize=(13, 8))
    plt.imshow(img, cmap="terrain", vmin=vmin, vmax=vmax)
    plt.colorbar(label="fase / altura relativa")
    plt.title(
        "Haz clic en este orden:\n"
        + "  ->  ".join(labels)
        + "\n\nCierra la ventana solo despues de terminar todos los clics."
    )

    pts = plt.ginput(len(labels), timeout=0)
    plt.close()

    if len(pts) != len(labels):
        raise RuntimeError(
            f"Se esperaban {len(labels)} clics, pero se recibieron {len(pts)}."
        )

    return {
        label: (int(round(x)), int(round(y)))
        for label, (x, y) in zip(labels, pts)
    }


def save_points(points, path):
    """Guarda puntos ROI a JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {
        k: {"x": int(v[0]), "y": int(v[1])}
        for k, v in points.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def load_points(path):
    """Carga puntos ROI desde JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        k: (int(v["x"]), int(v["y"]))
        for k, v in data.items()
    }


def ask_yes_no(question, default="y"):
    """Pregunta simple si/no."""
    default = default.lower().strip()
    suffix = "[Y/n]" if default == "y" else "[y/N]"

    ans = input(f"{question} {suffix}: ").strip().lower()

    if ans == "":
        ans = default

    return ans in ("y", "yes", "s", "si", "sí")


def roi_median(arr, mask, x, y, r=8):
    """
    Extrae la mediana en una ROI centrada en (x, y),
    usando solo pixeles validos de la mascara.
    """
    h, w = arr.shape

    if x < 0 or x >= w or y < 0 or y >= h:
        return np.nan

    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)

    roi = arr[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1] & np.isfinite(roi)

    if m.sum() < 5:
        return np.nan

    return float(np.median(roi[m]))


def metrics(pred, real):
    """Calcula metricas de error."""
    pred = np.asarray(pred, dtype=float)
    real = np.asarray(real, dtype=float)

    ok = np.isfinite(pred) & np.isfinite(real)
    if ok.sum() == 0:
        return {
            "N": 0,
            "MAE_cm": np.nan,
            "RMSE_cm": np.nan,
            "Bias_cm": np.nan,
            "MaxAbs_cm": np.nan,
            "MAPE_%": np.nan,
        }

    err = pred[ok] - real[ok]

    return {
        "N": int(ok.sum()),
        "MAE_cm": float(np.mean(np.abs(err))),
        "RMSE_cm": float(np.sqrt(np.mean(err ** 2))),
        "Bias_cm": float(np.mean(err)),
        "MaxAbs_cm": float(np.max(np.abs(err))),
        "MAPE_%": float(np.mean(np.abs(err / real[ok])) * 100.0),
    }


def fit_linear_cm(z_phase, z_real):
    """
    Ajuste empirico:
        z_cm = a * z_phase + b
    """
    z_phase = np.asarray(z_phase, dtype=float)
    z_real = np.asarray(z_real, dtype=float)

    ok = np.isfinite(z_phase) & np.isfinite(z_real)

    if ok.sum() < 2:
        raise RuntimeError("No hay suficientes puntos validos para calibracion lineal.")

    A = np.column_stack([z_phase[ok], np.ones(ok.sum())])
    a, b = np.linalg.lstsq(A, z_real[ok], rcond=None)[0]

    return float(a), float(b)


def leave_one_out_linear(z_phase, z_real):
    """
    Error mas honesto:
    para cada punto, calibra con todos los demas y predice el excluido.
    """
    z_phase = np.asarray(z_phase, dtype=float)
    z_real = np.asarray(z_real, dtype=float)

    pred = np.full_like(z_real, np.nan, dtype=float)

    for i in range(len(z_real)):
        train = np.ones(len(z_real), dtype=bool)
        train[i] = False

        train &= np.isfinite(z_phase) & np.isfinite(z_real)

        if train.sum() < 2 or not np.isfinite(z_phase[i]):
            pred[i] = np.nan
            continue

        a, b = fit_linear_cm(z_phase[train], z_real[train])
        pred[i] = a * z_phase[i] + b

    return pred


def estimate_period_cm_from_points(phi_pts, z_real, B=15.0, L=142.0):
    """
    Estima el periodo fisico P_cm usando la geometria:
        s = B*z/(L-z)
        P = 2*pi*s / |phi|

    Donde:
        s   = corrimiento horizontal fisico en pantalla, cm
        phi = fase medida, rad
        P   = periodo fisico de una franja, cm/franja
    """
    phi_pts = np.asarray(phi_pts, dtype=float)
    z_real = np.asarray(z_real, dtype=float)

    s_real = B * z_real / (L - z_real)

    ok = (
        np.isfinite(phi_pts)
        & np.isfinite(z_real)
        & (np.abs(phi_pts) > 1e-6)
        & (z_real > 0)
        & (z_real < L)
    )

    p_each = np.full_like(z_real, np.nan, dtype=float)
    p_each[ok] = 2.0 * np.pi * s_real[ok] / np.abs(phi_pts[ok])

    if np.isfinite(p_each).sum() == 0:
        return np.nan, p_each

    p_cm = float(np.nanmedian(p_each))

    return p_cm, p_each


def phase_to_height_geometry(phi, p_cm, B=15.0, L=142.0, sign=1.0):
    """
    Convierte fase a altura/profundidad en cm usando:
        s = phi * P / 2pi
        z = L*s/(B+s)

    sign corrige la convencion de fase.
    """
    if not np.isfinite(p_cm) or p_cm <= 0:
        return np.full_like(phi, np.nan, dtype=float)

    s = sign * phi * p_cm / (2.0 * np.pi)

    # Solo altura positiva hacia camara/proyector.
    s = np.where(s > 0, s, np.nan)

    z = L * s / (B + s)

    return z


def print_metrics_block(name, m):
    """Imprime metricas de forma legible."""
    print(f"\n{name}")
    print(f"  N       : {m['N']}")
    print(f"  MAE     : {m['MAE_cm']:.4f} cm")
    print(f"  RMSE    : {m['RMSE_cm']:.4f} cm")
    print(f"  Bias    : {m['Bias_cm']:.4f} cm")
    print(f"  MaxAbs  : {m['MaxAbs_cm']:.4f} cm")
    print(f"  MAPE    : {m['MAPE_%']:.2f} %")


def annotate_points_image(base_img, points, path):
    """Guarda una imagen con los puntos seleccionados."""
    finite = np.isfinite(base_img)
    if finite.any():
        vmin, vmax = np.nanpercentile(base_img[finite], [2, 98])
    else:
        vmin, vmax = None, None

    plt.figure(figsize=(12, 8))
    plt.imshow(base_img, cmap="terrain", vmin=vmin, vmax=vmax)
    plt.colorbar(label="fase / altura relativa")

    for label, (x, y) in points.items():
        plt.scatter([x], [y], s=35)
        plt.text(x + 5, y + 5, label, fontsize=8)

    plt.title("Puntos ROI seleccionados")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nCargando resultados...")
    z_phase_mf, z_phase_sf, mask_clean = load_results()

    labels = list(Z_REAL_CM.keys())
    z_real = np.array([Z_REAL_CM[k] for k in labels], dtype=float)

    display_img = np.where(mask_clean, z_phase_mf, np.nan)

    # --------------------------------------------------------
    # Seleccion / carga de puntos ROI
    # --------------------------------------------------------
    if os.path.exists(POINTS_JSON):
        use_saved = ask_yes_no(
            f"\nYa existe {POINTS_JSON}. ¿Quieres usar esos puntos?",
            default="y",
        )
    else:
        use_saved = False

    if use_saved:
        roi_points = load_points(POINTS_JSON)

        missing_labels = [k for k in labels if k not in roi_points]
        if missing_labels:
            print("\nEl JSON no tiene todos los puntos. Faltan:")
            for k in missing_labels:
                print(f"  - {k}")
            print("Se volveran a seleccionar todos los puntos.")
            roi_points = pick_rois(display_img, labels)
            save_points(roi_points, POINTS_JSON)
    else:
        print("\nSelecciona puntos con clic.")
        print("Orden:")
        for i, label in enumerate(labels, start=1):
            print(f"  {i:02d}. {label}")

        roi_points = pick_rois(display_img, labels)
        save_points(roi_points, POINTS_JSON)
        print(f"\nPuntos guardados en: {POINTS_JSON}")

    annotate_points_image(
        display_img,
        roi_points,
        os.path.join(OUT_DIR, "roi_points_selected.png"),
    )

    # --------------------------------------------------------
    # Extraer valores de fase en cada ROI
    # --------------------------------------------------------
    z_phase_mf_pts = np.array([
        roi_median(z_phase_mf, mask_clean, *roi_points[k], r=ROI_HALF)
        for k in labels
    ], dtype=float)

    z_phase_sf_pts = np.array([
        roi_median(z_phase_sf, mask_clean, *roi_points[k], r=ROI_HALF)
        for k in labels
    ], dtype=float)

    valid_count_mf = np.isfinite(z_phase_mf_pts).sum()
    valid_count_sf = np.isfinite(z_phase_sf_pts).sum()

    print(f"\nPuntos validos MF: {valid_count_mf}/{len(labels)}")
    print(f"Puntos validos SF: {valid_count_sf}/{len(labels)}")

    if valid_count_mf < 2:
        raise RuntimeError(
            "Muy pocos puntos validos en multi-freq. "
            "Revisa la mascara o aumenta ROI_HALF."
        )

    if valid_count_sf < 2:
        print(
            "\nADVERTENCIA: Muy pocos puntos validos en single-freq. "
            "Se omitira parte de la comparacion SF."
        )

    # --------------------------------------------------------
    # A) Calibracion lineal empirica
    # --------------------------------------------------------
    a_mf, b_mf = fit_linear_cm(z_phase_mf_pts, z_real)
    z_pred_mf_linear = a_mf * z_phase_mf_pts + b_mf
    z_pred_mf_loo = leave_one_out_linear(z_phase_mf_pts, z_real)
    z_mf_cm_linear_map = a_mf * z_phase_mf + b_mf

    if valid_count_sf >= 2:
        a_sf, b_sf = fit_linear_cm(z_phase_sf_pts, z_real)
        z_pred_sf_linear = a_sf * z_phase_sf_pts + b_sf
        z_pred_sf_loo = leave_one_out_linear(z_phase_sf_pts, z_real)
        z_sf_cm_linear_map = a_sf * z_phase_sf + b_sf
    else:
        a_sf, b_sf = np.nan, np.nan
        z_pred_sf_linear = np.full_like(z_real, np.nan)
        z_pred_sf_loo = np.full_like(z_real, np.nan)
        z_sf_cm_linear_map = np.full_like(z_phase_sf, np.nan)

    # --------------------------------------------------------
    # B) Calibracion geometrica aproximada
    # --------------------------------------------------------
    p_cm_mf, p_each_mf = estimate_period_cm_from_points(
        z_phase_mf_pts,
        z_real,
        B=B_CM,
        L=L_CM,
    )

    # Signo de fase:
    # Si la mayoria de los puntos medidos tienen fase negativa,
    # se invierte para obtener altura positiva.
    sign_mf = np.sign(np.nanmedian(z_phase_mf_pts))
    if not np.isfinite(sign_mf) or sign_mf == 0:
        sign_mf = 1.0

    z_mf_cm_geom_map = phase_to_height_geometry(
        z_phase_mf,
        p_cm=p_cm_mf,
        B=B_CM,
        L=L_CM,
        sign=sign_mf,
    )

    z_pred_mf_geom = np.array([
        roi_median(z_mf_cm_geom_map, mask_clean, *roi_points[k], r=ROI_HALF)
        for k in labels
    ], dtype=float)

    # --------------------------------------------------------
    # Tabla de resultados
    # --------------------------------------------------------
    df = pd.DataFrame({
        "Punto": labels,
        "x_px": [roi_points[k][0] for k in labels],
        "y_px": [roi_points[k][1] for k in labels],
        "z_real_cm": z_real,

        "z_phase_mf_rad": z_phase_mf_pts,
        "z_mf_linear_cm": z_pred_mf_linear,
        "error_mf_linear_cm": z_pred_mf_linear - z_real,

        "z_mf_LOO_cm": z_pred_mf_loo,
        "error_mf_LOO_cm": z_pred_mf_loo - z_real,

        "periodo_estimado_por_punto_cm": p_each_mf,
        "z_mf_geom_cm": z_pred_mf_geom,
        "error_mf_geom_cm": z_pred_mf_geom - z_real,

        "z_phase_sf_rad": z_phase_sf_pts,
        "z_sf_linear_cm": z_pred_sf_linear,
        "error_sf_linear_cm": z_pred_sf_linear - z_real,

        "z_sf_LOO_cm": z_pred_sf_loo,
        "error_sf_LOO_cm": z_pred_sf_loo - z_real,
    })

    csv_path = os.path.join(OUT_DIR, "error_analysis.csv")
    df.to_csv(csv_path, index=False)

    # --------------------------------------------------------
    # Imprimir resumen
    # --------------------------------------------------------
    print("\n============================================================")
    print("PARAMETROS CALIBRADOS")
    print("============================================================")
    print(f"MF lineal: z_cm = {a_mf:.6f} * fase + {b_mf:.6f}")

    if np.isfinite(a_sf):
        print(f"SF lineal: z_cm = {a_sf:.6f} * fase + {b_sf:.6f}")
    else:
        print("SF lineal: no disponible")

    print(f"P_cm MF geometrico estimado = {p_cm_mf:.6f} cm/franja")
    print(f"Signo fase MF = {sign_mf:+.0f}")

    print("\n============================================================")
    print("METRICAS")
    print("============================================================")

    print_metrics_block(
        "MF lineal, ajustando y evaluando con los mismos puntos",
        metrics(z_pred_mf_linear, z_real),
    )

    print_metrics_block(
        "MF Leave-One-Out, mas honesto",
        metrics(z_pred_mf_loo, z_real),
    )

    print_metrics_block(
        "MF geometrico",
        metrics(z_pred_mf_geom, z_real),
    )

    if valid_count_sf >= 2:
        print_metrics_block(
            "SF lineal, ajustando y evaluando con los mismos puntos",
            metrics(z_pred_sf_linear, z_real),
        )

        print_metrics_block(
            "SF Leave-One-Out",
            metrics(z_pred_sf_loo, z_real),
        )

    print("\n============================================================")
    print("TABLA DE ERRORES")
    print("============================================================")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df)

    # --------------------------------------------------------
    # Graficas
    # --------------------------------------------------------

    # Real vs predicho
    plt.figure(figsize=(7, 7))
    plt.scatter(df["z_real_cm"], df["z_mf_linear_cm"], label="MF lineal")
    plt.scatter(df["z_real_cm"], df["z_mf_geom_cm"], label="MF geom")

    if valid_count_sf >= 2:
        plt.scatter(df["z_real_cm"], df["z_sf_linear_cm"], label="SF lineal")

    pred_cols = ["z_mf_linear_cm", "z_mf_geom_cm"]
    if valid_count_sf >= 2:
        pred_cols.append("z_sf_linear_cm")

    mn = np.nanmin([df["z_real_cm"].min(), df[pred_cols].min().min()])
    mx = np.nanmax([df["z_real_cm"].max(), df[pred_cols].max().max()])

    plt.plot([mn, mx], [mn, mx], linestyle="--", label="ideal")
    plt.xlabel("Z real [cm]")
    plt.ylabel("Z reconstruida [cm]")
    plt.title("Reconstruccion vs medicion real")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "z_real_vs_pred.png"), dpi=120)
    plt.close()

    # Error por punto MF LOO
    plt.figure(figsize=(10, 5))
    plt.bar(df["Punto"], df["error_mf_LOO_cm"])
    plt.axhline(0, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Error [cm]")
    plt.title("Error MF Leave-One-Out por punto")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "error_mf_loo_por_punto.png"), dpi=120)
    plt.close()

    # Error por punto MF lineal
    plt.figure(figsize=(10, 5))
    plt.bar(df["Punto"], df["error_mf_linear_cm"])
    plt.axhline(0, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Error [cm]")
    plt.title("Error MF lineal por punto")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "error_mf_linear_por_punto.png"), dpi=120)
    plt.close()

    # Error por punto MF geometrico
    plt.figure(figsize=(10, 5))
    plt.bar(df["Punto"], df["error_mf_geom_cm"])
    plt.axhline(0, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Error [cm]")
    plt.title("Error MF geometrico por punto")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "error_mf_geom_por_punto.png"), dpi=120)
    plt.close()

    # Mapas calibrados
    save_map(
        np.where(mask_clean, z_mf_cm_linear_map, np.nan),
        os.path.join(OUT_DIR, "Z_MF_cm_linear.png"),
        "Altura Z multi-freq calibrada linealmente [cm]",
        "terrain",
    )

    save_map(
        np.where(mask_clean, z_mf_cm_geom_map, np.nan),
        os.path.join(OUT_DIR, "Z_MF_cm_geometry.png"),
        "Altura Z multi-freq calibrada geometricamente [cm]",
        "terrain",
    )

    if valid_count_sf >= 2:
        save_map(
            np.where(mask_clean, z_sf_cm_linear_map, np.nan),
            os.path.join(OUT_DIR, "Z_SF_cm_linear.png"),
            "Altura Z single-freq calibrada linealmente [cm]",
            "terrain",
        )

    # Guardar matrices numericas
    np.savez(
        os.path.join(OUT_DIR, "result_error_analysis.npz"),
        Z_MF_CM_LINEAR=z_mf_cm_linear_map,
        Z_MF_CM_GEOM=z_mf_cm_geom_map,
        Z_SF_CM_LINEAR=z_sf_cm_linear_map,
        z_real=z_real,
        z_phase_mf=z_phase_mf_pts,
        z_phase_sf=z_phase_sf_pts,
        labels=np.array(labels),
        x_px=np.array([roi_points[k][0] for k in labels]),
        y_px=np.array([roi_points[k][1] for k in labels]),
        a_mf=a_mf,
        b_mf=b_mf,
        a_sf=a_sf,
        b_sf=b_sf,
        p_cm_mf=p_cm_mf,
        sign_mf=sign_mf,
    )

    print("\n============================================================")
    print("ARCHIVOS GUARDADOS")
    print("============================================================")
    print(f"CSV errores: {csv_path}")
    print(f"Puntos ROI: {POINTS_JSON}")
    print(f"Imagen puntos: {os.path.join(OUT_DIR, 'roi_points_selected.png')}")
    print(f"Grafica real vs pred: {os.path.join(OUT_DIR, 'z_real_vs_pred.png')}")
    print(f"Mapa MF cm lineal: {os.path.join(OUT_DIR, 'Z_MF_cm_linear.png')}")
    print(f"Mapa MF cm geometrico: {os.path.join(OUT_DIR, 'Z_MF_cm_geometry.png')}")
    print(f"NPZ completo: {os.path.join(OUT_DIR, 'result_error_analysis.npz')}")
    print("\nListo.")


if __name__ == "__main__":
    main()
