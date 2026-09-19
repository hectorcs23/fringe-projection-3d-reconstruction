"""
Self-contained demo of the multi-frequency phase-shifting pipeline.

What it does:
  1. Builds a synthetic 3D test object (a small "landscape" with bumps,
     edges, and a hole).
  2. Simulates projecting 4 fringe frequencies (1, 4, 16, 64 fringes)
     each with 4 phase shifts = 16 camera images of the object.
  3. Does the same without the object -> 16 reference-plane images.
  4. Saves all 32 images as PNGs into ./synth_images/.
  5. Runs the full pipeline with multi-frequency phase unwrapping.
  6. Meshes the result and writes surface_from_synth.stl.

Run this first to confirm everything works on your machine. Then replace
the image-synthesis step with your real capture, keeping the same file
naming convention.

Requires: numpy, scikit-image, Pillow, open3d, and SuperficieSTL.py
          in the same folder.
"""
import os
import numpy as np
from PIL import Image

from phase_shifting import multifreq_images_to_heightmap


# ---------------- Ground-truth object ----------------

def make_test_object(ny=400, nx=400):
    """A little landscape: two gaussians, a plateau, and a hole."""
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    Z = (
          12.0 * np.exp(-((X - 130) ** 2 + (Y - 150) ** 2) / (2 * 40 ** 2))
        +  8.0 * np.exp(-((X - 260) ** 2 + (Y - 120) ** 2) / (2 * 25 ** 2))
    )
    # A flat plateau on the right side
    plateau = (X > 280) & (X < 360) & (Y > 220) & (Y < 320)
    Z[plateau] += 6.0

    # A hole in the background (invalid pixels)
    hole = ((X - 80) ** 2 + (Y - 300) ** 2) < 20 ** 2
    mask = np.ones_like(Z, dtype=bool)
    mask[hole] = False
    return Z, mask


# ---------------- Fringe synthesis ----------------

def synth_fringes(Z, N_fringes, N_shifts=4, K_height=0.25,
                  A=120.0, B=60.0, noise_sigma=0.8,
                  axis="vertical"):
    """
    Generate N_shifts phase-shifted camera images at a given fringe frequency.

    axis: "vertical" -> stripes run horizontally (phase varies along y),
                        which matches a vertical projector-camera baseline.
    N_fringes   : how many full fringes appear across the image.
    K_height    : radians of phase per mm of height at the HIGHEST frequency.
                  We scale with N_fringes so that taller objects = more
                  phase shift, consistently across frequencies.
    """
    ny, nx = Z.shape
    j, i = np.meshgrid(np.arange(nx), np.arange(ny))
    if axis == "vertical":
        carrier = 2.0 * np.pi * N_fringes * (i / ny)
    else:
        carrier = 2.0 * np.pi * N_fringes * (j / nx)
    phi_true = carrier + K_height * N_fringes * Z  # height phase scales with freq

    imgs = []
    for k in range(N_shifts):
        shift = 2.0 * np.pi * k / N_shifts
        I = A + B * np.cos(phi_true + shift)
        I += noise_sigma * np.random.randn(ny, nx)
        I = np.clip(I, 0.0, 255.0)
        imgs.append(I.astype(np.uint8))
    return imgs


# ---------------- Save / group helpers ----------------

def save_group(imgs, folder, prefix):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for k, im in enumerate(imgs):
        p = os.path.join(folder, f"{prefix}_{k}.png")
        Image.fromarray(im).save(p)
        paths.append(p)
    return paths


# ---------------- Main ----------------

def main():
    np.random.seed(0)
    folder = "synth_images"

    print("Building ground-truth object ...")
    Z_true, mask_true = make_test_object(ny=400, nx=400)
    # Apply the hole: outside the mask, set Z to 0 (flat reference)
    Z_object = np.where(mask_true, Z_true, 0.0)
    Z_reference = np.zeros_like(Z_true)

    fringe_counts = [1, 4, 16, 64]
    N_shifts = 4
    K_HEIGHT_PER_RAD = 1.0 / (0.25 * 64)  # inverse of K_height*N_highest

    print(f"Synthesizing {len(fringe_counts)} frequencies x "
          f"{N_shifts} shifts x 2 (object+reference) "
          f"= {len(fringe_counts) * N_shifts * 2} images ...")

    object_groups, reference_groups = [], []
    for N_f in fringe_counts:
        obj_imgs = synth_fringes(Z_object,    N_fringes=N_f, N_shifts=N_shifts)
        ref_imgs = synth_fringes(Z_reference, N_fringes=N_f, N_shifts=N_shifts)
        object_groups.append(save_group(obj_imgs, folder, f"obj_f{N_f}"))
        reference_groups.append(save_group(ref_imgs, folder, f"ref_f{N_f}"))

    print(f"Saved images to ./{folder}/")
    print("Running multi-frequency pipeline ...")

    X, Y, Z_rec, mask = multifreq_images_to_heightmap(
        object_groups=object_groups,
        fringe_counts=fringe_counts,
        reference_groups=reference_groups,
        pixel_pitch_x=1.0,
        pixel_pitch_y=1.0,
        K=K_HEIGHT_PER_RAD,
    )

    err = (Z_rec - Z_true)[mask & mask_true]
    print(f"  Z_true range     = [{Z_true.min():.2f}, {Z_true.max():.2f}] mm")
    print(f"  Z_recovered rng  = [{Z_rec[mask].min():.2f}, {Z_rec[mask].max():.2f}] mm")
    print(f"  max |error|      = {np.abs(err).max():.3f} mm")
    print(f"  rms  error       = {np.sqrt((err ** 2).mean()):.3f} mm")
    print(f"  valid pixels     = {mask.mean() * 100:.1f} %")

    # Mesh + export using the user's existing pipeline
    try:
        from SuperficieSTL import heightmap_to_mesh, smooth_mesh, export_stl
        import open3d as o3d
        print("Meshing ...")
        mesh = heightmap_to_mesh(X, Y, Z_rec, mask)
        mesh = smooth_mesh(mesh, iterations=3)
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        export_stl(mesh, "surface_from_synth.stl")
        # Uncomment next line to open the viewer:
        # o3d.visualization.draw_geometries([mesh],
        #     mesh_show_back_face=True, mesh_show_wireframe=True)
    except ImportError as e:
        print(f"  (meshing skipped: {e})")


if __name__ == "__main__":
    main()
