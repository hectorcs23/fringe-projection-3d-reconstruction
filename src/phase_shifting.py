"""
Phase-shifting profilometry pipeline.

Takes N phase-shifted images of a fringe pattern projected onto an object
and produces a height map Z(x, y) + mask suitable for the heightmap_to_mesh
function in SuperficieSTL.py.

Pipeline:
  1. N-step phase shifting -> wrapped phase, modulation, DC term
  2. Quality mask from modulation and saturation
  3. 2D phase unwrapping (skimage.restoration.unwrap_phase)
  4. Remove reference-plane phase (from reference images or fitted plane)
  5. Linear phase-to-height via calibration constant K

Required packages: numpy, scikit-image, and (cv2 or Pillow) for image IO.
"""
from __future__ import annotations
import numpy as np


# ---------------- Image loading ----------------

def load_gray(path):
    """Load an image as a 2D float64 grayscale array. Uses cv2 if present, else PIL."""
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not read {path}")
    except ImportError:
        from PIL import Image
        img = np.asarray(Image.open(path).convert("L"))
    return img.astype(np.float64)


# ---------------- Core phase-shifting math ----------------

def n_step_phase(images):
    """
    Generic N-step phase shifting with equal shifts delta_k = 2*pi*k/N.

    Given  I_k = A + B * cos(phi + delta_k),  for k = 0, ..., N-1 with N >= 3:

        sum_k I_k * cos(delta_k) = (N/2) * B * cos(phi)
        sum_k I_k * sin(delta_k) = -(N/2) * B * sin(phi)

    so   phi = atan2(-sum I_k sin delta_k, sum I_k cos delta_k).

    Parameters
    ----------
    images : sequence of N 2D arrays (same shape) in order of increasing shift.

    Returns
    -------
    phi_wrapped : 2D array in [-pi, pi]
    modulation  : 2D array, equals B (fringe contrast per pixel)
    average     : 2D array, equals A (DC / ambient per pixel)
    """
    imgs = np.stack([np.asarray(im, dtype=np.float64) for im in images], axis=0)
    N = imgs.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 phase-shifted images.")

    k = np.arange(N)
    delta = 2.0 * np.pi * k / N
    cos_d = np.cos(delta)[:, None, None]
    sin_d = np.sin(delta)[:, None, None]

    num = -np.sum(imgs * sin_d, axis=0)   # proportional to  B sin(phi)
    den =  np.sum(imgs * cos_d, axis=0)   # proportional to  B cos(phi)

    phi_wrapped = np.arctan2(num, den)
    modulation  = (2.0 / N) * np.sqrt(num * num + den * den)
    average     = imgs.mean(axis=0)
    return phi_wrapped, modulation, average


# ---------------- Data-quality mask ----------------

def build_mask(modulation, average,
               mod_frac=0.1, dc_min=5.0, dc_max=250.0):
    """
    Valid pixels: enough fringe contrast and neither underexposed nor saturated.
    mod_frac is a fraction of the 99th percentile of modulation (robust to hot spots).
    """
    mod_threshold = mod_frac * np.percentile(modulation, 99)
    mask  = modulation > mod_threshold
    mask &= average > dc_min
    mask &= average < dc_max
    return mask


# ---------------- Phase unwrapping ----------------

def unwrap_2d(phi_wrapped, mask=None):
    """
    2D spatial phase unwrapping via scikit-image. Masked pixels are
    skipped by the unwrapper and then filled with 0 so arithmetic works;
    the mask should still be passed through to the mesher so those
    pixels don't get triangulated.
    """
    from skimage.restoration import unwrap_phase
    if mask is None:
        return np.asarray(unwrap_phase(phi_wrapped))
    ma = np.ma.array(phi_wrapped, mask=~mask)
    unwrapped = unwrap_phase(ma)
    return np.ma.filled(unwrapped, 0.0)


def multi_frequency_unwrap(wrapped_phases, fringe_counts, mask=None):
    """
    Hierarchical temporal phase unwrapping from multiple fringe frequencies.

    Much more robust than spatial unwrap: each pixel is unwrapped using
    only data from that pixel, so edges / holes / discontinuities don't
    cause error propagation.

    The idea: at the coarsest frequency N_1 (few fringes across the image),
    spatial unwrap is trivial. The unwrapped coarse phase tells you the
    integer fringe order k_2 at each pixel for the next frequency N_2:

        Phi_2_expected = (N_2/N_1) * Phi_1
        k_2            = round((Phi_2_expected - phi_2_wrapped) / (2*pi))
        Phi_2          = phi_2_wrapped + 2*pi*k_2

    Then Phi_2 unwraps Phi_3, and so on up to the highest frequency.
    The final Phi_M is unambiguous AND uses the high-frequency precision.

    Parameters
    ----------
    wrapped_phases : list of M 2D arrays, wrapped phase maps in [-pi, pi],
                     sorted from lowest to highest fringe frequency.
    fringe_counts  : list of M positive numbers (N_1 < N_2 < ... < N_M).
                     The number of fringes projected at each frequency.
    mask           : optional boolean mask of valid pixels.

    Returns
    -------
    Phi_unwrapped : 2D array. Unwrapped absolute phase at the highest
                    frequency. Proportional to the projector coordinate.
    """
    phases = [np.asarray(p, dtype=np.float64) for p in wrapped_phases]
    N = np.asarray(fringe_counts, dtype=np.float64)
    if len(phases) != len(N):
        raise ValueError("Need one fringe count per wrapped-phase map.")
    if not np.all(np.diff(N) > 0):
        raise ValueError("fringe_counts must be strictly increasing.")

    # Coarsest frequency: spatial unwrap (safe because it wraps few times).
    Phi = unwrap_2d(phases[0], mask=mask)

    # Walk up the ladder
    for k in range(1, len(phases)):
        ratio = N[k] / N[k - 1]
        Phi_expected = ratio * Phi
        order = np.round((Phi_expected - phases[k]) / (2.0 * np.pi))
        Phi = phases[k] + 2.0 * np.pi * order

    return Phi


def multi_frequency_unwrap_robust(wrapped_phases, fringe_counts, mask=None,
                                  median_size=5):
    """
    Versión robusta: aplica filtro de mediana al orden de franja en cada
    nivel para eliminar outliers aislados antes de pasar al siguiente.
    Esto es crucial cuando el ratio entre frecuencias es grande (>3)
    o cuando las fases de frecuencias bajas tienen ruido.

    median_size : ventana del filtro de mediana aplicado al orden entero.
                  3-5 es suficiente. El filtro solo corrige outliers
                  aislados, no puede arreglar bandas grandes de error.
    """
    from scipy.ndimage import median_filter

    phases = [np.asarray(p, dtype=np.float64) for p in wrapped_phases]
    N = np.asarray(fringe_counts, dtype=np.float64)
    if len(phases) != len(N):
        raise ValueError("Need one fringe count per wrapped-phase map.")

    Phi = unwrap_2d(phases[0], mask=mask)

    for k in range(1, len(phases)):
        ratio = N[k] / N[k - 1]
        Phi_expected = ratio * Phi
        order = np.round((Phi_expected - phases[k]) / (2.0 * np.pi))
        # Filtro de mediana al orden entero: corrige saltos aislados
        order = median_filter(order, size=median_size)
        Phi = phases[k] + 2.0 * np.pi * order

    return Phi


# ---------------- Plane detrending (no-reference fallback) ----------------

def subtract_fitted_plane(phi, mask, n_trim=3, keep_frac=0.5):
    """
    Iteratively fit phi ~ a*i + b*j + c on the valid pixels and subtract it.
    Removes the carrier fringe (the linear phase ramp produced by the
    projector on a flat reference) so what is left is the object phase.

    Robustness: after each fit, we keep only the valid pixels with the
    smallest |residual| for the next fit. This prevents tall features from
    biasing the plane. keep_frac = fraction kept per iteration.
    """
    ny, nx = phi.shape
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))

    active = mask.copy()
    for _ in range(max(1, n_trim)):
        idx = active.ravel()
        A = np.column_stack([ii.ravel()[idx],
                             jj.ravel()[idx],
                             np.ones(idx.sum())])
        b = phi.ravel()[idx]
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        a_, b_, c_ = coeffs
        plane = a_ * ii + b_ * jj + c_
        resid = np.abs(phi - plane)
        cutoff = np.quantile(resid[active], keep_frac)
        active = mask & (resid <= cutoff)

    return phi - plane


def wrap_to_pi(x):
    """Wrap an array of angles into (-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def anchor_background(Z, mask, percentile=5.0):
    """
    Shift Z so that its `percentile`-th percentile on valid pixels is zero.
    Assumes the lowest region in view is the reference plane / background.
    """
    return Z - np.percentile(Z[mask], percentile)


# ---------------- Phase -> height ----------------

def phase_to_height(delta_phi, K=1.0):
    """
    Linear conversion h = K * delta_phi.

    K is a calibration constant with units of length-per-radian.
    Calibrate it by imaging an object of known height h0 and reading
    the recovered delta_phi at that point: K = h0 / delta_phi.

    For higher accuracy, replace this with the full geometric relation:
        h = L * delta_phi / (delta_phi - 2*pi*d/p)
    where L = reference-plane to camera distance, d = projector-camera
    baseline, p = fringe period on the reference plane.
    """
    return K * delta_phi


# ---------------- End-to-end wrappers ----------------

def images_to_heightmap(object_paths, reference_paths=None,
                        pixel_pitch_x=1.0, pixel_pitch_y=1.0, K=1.0,
                        mod_frac=0.1):
    """
    Single-frequency pipeline: N phase-shifted images -> (X, Y, Z, mask).

    Fine for small, smooth objects with no discontinuities. For anything
    real, prefer multifreq_images_to_heightmap().
    """
    imgs_obj = [load_gray(p) for p in object_paths]
    phi_o_w, mod_o, dc_o = n_step_phase(imgs_obj)
    mask = build_mask(mod_o, dc_o, mod_frac=mod_frac)

    if reference_paths is not None:
        imgs_ref = [load_gray(p) for p in reference_paths]
        phi_r_w, mod_r, dc_r = n_step_phase(imgs_ref)
        mask &= build_mask(mod_r, dc_r, mod_frac=mod_frac)
        delta_phi = unwrap_2d(wrap_to_pi(phi_o_w - phi_r_w), mask)
    else:
        phi_o_u = unwrap_2d(phi_o_w, mask)
        delta_phi = subtract_fitted_plane(phi_o_u, mask)

    Z = phase_to_height(delta_phi, K=K)
    Z = anchor_background(Z, mask, percentile=5.0)
    ny, nx = Z.shape
    x = np.arange(nx) * pixel_pitch_x
    y = np.arange(ny) * pixel_pitch_y
    X, Y = np.meshgrid(x, y)
    return X, Y, Z, mask


def multifreq_images_to_heightmap(object_groups, fringe_counts,
                                  reference_groups=None,
                                  pixel_pitch_x=1.0, pixel_pitch_y=1.0,
                                  K=1.0, mod_frac=0.1):
    """
    Multi-frequency pipeline. This is the recommended function.

    Parameters
    ----------
    object_groups   : list of M lists, one per fringe frequency.
                      object_groups[k] is a list of N phase-shifted image
                      paths at frequency k (k = 0 = lowest, M-1 = highest).
    fringe_counts   : list of M numbers giving number of fringes projected
                      at each frequency (strictly increasing).
    reference_groups: same structure as object_groups, for the flat
                      reference plane. Strongly recommended. If None,
                      a robust plane fit is used on the object phase.
    pixel_pitch_x/y : mm per pixel on the object.
    K               : phase-to-height constant (length per radian) at the
                      HIGHEST frequency. You calibrate K the same way:
                      image something of known height, run with K=1, and
                      set K = h_known / recovered_Z_known.

    Returns (X, Y, Z, mask) -- same format as create_test_surface().
    """
    M = len(object_groups)
    if len(fringe_counts) != M:
        raise ValueError("Need one fringe count per frequency group.")

    # Compute wrapped phase + modulation + DC at each frequency
    phi_o_w, mod_o, dc_o = [], [], []
    for paths in object_groups:
        imgs = [load_gray(p) for p in paths]
        pw, mo, dc = n_step_phase(imgs)
        phi_o_w.append(pw)
        mod_o.append(mo)
        dc_o.append(dc)

    # Mask: require good modulation at the highest frequency (it dominates
    # precision), and sane exposure at every frequency.
    mask = build_mask(mod_o[-1], dc_o[-1], mod_frac=mod_frac)
    for dc in dc_o[:-1]:
        mask &= (dc > 5.0) & (dc < 250.0)

    Phi_obj = multi_frequency_unwrap(phi_o_w, fringe_counts, mask=mask)

    if reference_groups is not None:
        phi_r_w, mod_r, dc_r = [], [], []
        for paths in reference_groups:
            imgs = [load_gray(p) for p in paths]
            pw, mo, dc = n_step_phase(imgs)
            phi_r_w.append(pw)
            mod_r.append(mo)
            dc_r.append(dc)
        mask &= build_mask(mod_r[-1], dc_r[-1], mod_frac=mod_frac)
        Phi_ref = multi_frequency_unwrap(phi_r_w, fringe_counts, mask=mask)
        delta_phi = Phi_obj - Phi_ref
    else:
        delta_phi = subtract_fitted_plane(Phi_obj, mask)

    Z = phase_to_height(delta_phi, K=K)
    Z = anchor_background(Z, mask, percentile=5.0)
    ny, nx = Z.shape
    x = np.arange(nx) * pixel_pitch_x
    y = np.arange(ny) * pixel_pitch_y
    X, Y = np.meshgrid(x, y)
    return X, Y, Z, mask

    ny, nx = Z.shape
    x = np.arange(nx) * pixel_pitch_x
    y = np.arange(ny) * pixel_pitch_y
    X, Y = np.meshgrid(x, y)
    return X, Y, Z, mask


# ---------------- Arrays-in variant (if images are already in memory) ----------------

def arrays_to_heightmap(object_arrays, reference_arrays=None,
                        pixel_pitch_x=1.0, pixel_pitch_y=1.0, K=1.0,
                        mod_frac=0.1):
    """Same as images_to_heightmap but takes numpy arrays directly."""
    phi_o_w, mod_o, dc_o = n_step_phase(object_arrays)
    mask = build_mask(mod_o, dc_o, mod_frac=mod_frac)

    if reference_arrays is not None:
        phi_r_w, mod_r, dc_r = n_step_phase(reference_arrays)
        mask &= build_mask(mod_r, dc_r, mod_frac=mod_frac)
        delta_phi = unwrap_2d(wrap_to_pi(phi_o_w - phi_r_w), mask)
    else:
        phi_o_u = unwrap_2d(phi_o_w, mask)
        delta_phi = subtract_fitted_plane(phi_o_u, mask)

    Z = phase_to_height(delta_phi, K=K)
    Z = anchor_background(Z, mask, percentile=5.0)
    ny, nx = Z.shape
    x = np.arange(nx) * pixel_pitch_x
    y = np.arange(ny) * pixel_pitch_y
    X, Y = np.meshgrid(x, y)
    return X, Y, Z, mask


# ---------------- Example ----------------

if __name__ == "__main__":
    # Replace these paths with your captured images, in order of increasing shift.
    object_imgs = ["obj_0.png", "obj_1.png", "obj_2.png", "obj_3.png"]
    ref_imgs    = ["ref_0.png", "ref_1.png", "ref_2.png", "ref_3.png"]  # or None

    X, Y, Z, mask = images_to_heightmap(
        object_imgs,
        reference_paths=ref_imgs,   # set to None if no reference capture
        pixel_pitch_x=0.5,          # mm/pixel from camera calibration
        pixel_pitch_y=0.5,
        K=1.0,                      # mm per radian (calibrate this!)
    )

    # Feed straight into your existing pipeline:
    from SuperficieSTL import heightmap_to_mesh, smooth_mesh, export_stl
    import open3d as o3d

    mesh = heightmap_to_mesh(X, Y, Z, mask)
    mesh = smooth_mesh(mesh, iterations=3)
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries(
        [mesh],
        mesh_show_back_face=True,
        mesh_show_wireframe=True,
    )
    export_stl(mesh, "surface_from_photos.stl")
