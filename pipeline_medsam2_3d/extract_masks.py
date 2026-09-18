# -*- coding: utf-8 -*-
"""
extract_masks.py — Extract 2D segmentation mask slices from a NIfTI volume
along three orthogonal planes and save as ordered PNGs.

Views:
  - Axial   (top)   : vol[:, :, z]  — looking down from above
  - Coronal (front)  : vol[:, y, :]  — looking from the front
  - Sagittal (side)  : vol[x, :, :]  — looking from the side

Only non-empty slices (containing at least one foreground pixel) are saved,
unless SAVE_EMPTY is set to True.

Reorientation is applied per-view so slices appear in standard radiological
orientation (matching what you'd see in ITK-SNAP / 3D Slicer).

Usage:
    python extract_masks.py
    python extract_masks.py --nifti /path/to/file.nii.gz
    python extract_masks.py --nifti /path/to/file.nii.gz --output /path/to/output
"""

import argparse
import os
import sys
import time

import numpy as np
import nibabel as nib
from PIL import Image

# =======================================================================
# ========================= DEFAULT CONFIG ==============================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"

# Default NIfTI to extract (override with --nifti)
DEFAULT_NIFTI = os.path.join(
    PROJECT_ROOT, "Pat18segbonecomplete.nii"
)

# Default output root (override with --output)
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "pipeline_medsam2_3d/p18_extracted_masks")

# Save empty (all-zero) slices too?
SAVE_EMPTY = False

# =======================================================================
# ========================= HELPERS =====================================
# =======================================================================


def reorient_axial(slice_2d: np.ndarray) -> np.ndarray:
    """Reorient an axial slice for display.
    Rotate 90° CW and flip LR to match standard radiological view."""
    img = np.rot90(slice_2d, k=3)  # 270° CCW = 90° CW
    img = np.fliplr(img)
    return img


def reorient_coronal(slice_2d: np.ndarray) -> np.ndarray:
    """Reorient a coronal slice for display.
    Transpose so Z becomes vertical (rows), then flip so superior is at top."""
    img = slice_2d.T  # (H, Z) → (Z, H) — Z is now vertical
    img = np.flipud(img)  # superior at top
    return img


def reorient_sagittal(slice_2d: np.ndarray) -> np.ndarray:
    """Reorient a sagittal slice for display.
    Transpose so Z becomes vertical (rows), then flip so superior is at top."""
    img = slice_2d.T  # (W, Z) → (Z, W) — Z is now vertical
    img = np.flipud(img)  # superior at top
    return img


def save_slices(
    volume: np.ndarray,
    axis: int,
    axis_name: str,
    output_dir: str,
    reorient_fn,
    save_empty: bool = False,
):
    """
    Extract 2D slices along the given axis, reorient, and save as PNGs.

    Args:
        volume:      3D numpy array (H, W, Z)
        axis:        0=sagittal(x), 1=coronal(y), 2=axial(z)
        axis_name:   'axial', 'coronal', or 'sagittal'
        output_dir:  folder to save PNGs into
        reorient_fn: function to reorient the 2D slice for display
        save_empty:  if True, save all slices; if False, skip empty ones
    """
    os.makedirs(output_dir, exist_ok=True)
    n_slices = volume.shape[axis]
    saved = 0
    total_nonempty = 0

    for i in range(n_slices):
        # Extract slice along the given axis
        if axis == 0:
            sl = volume[i, :, :]
        elif axis == 1:
            sl = volume[:, i, :]
        else:
            sl = volume[:, :, i]

        has_content = np.any(sl > 0)
        if has_content:
            total_nonempty += 1

        if not save_empty and not has_content:
            continue

        # Reorient for display
        sl = reorient_fn(sl)

        # Convert to 0/255 uint8
        sl_img = (sl > 0).astype(np.uint8) * 255

        # Save with zero-padded index
        fname = f"{axis_name}_{i:04d}.png"
        Image.fromarray(sl_img, mode="L").save(os.path.join(output_dir, fname))
        saved += 1

    return saved, total_nonempty, n_slices


# =======================================================================
# ========================= MAIN =======================================
# =======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract 2D mask slices from a NIfTI volume (axial, coronal, sagittal)"
    )
    parser.add_argument(
        "--nifti", type=str, default=DEFAULT_NIFTI, help="Path to the input NIfTI file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Root output directory (subfolders axial/, coronal/, sagittal/ created inside)",
    )
    parser.add_argument(
        "--save-empty",
        action="store_true",
        default=SAVE_EMPTY,
        help="Also save empty (all-zero) slices",
    )
    args = parser.parse_args()

    nifti_path = args.nifti
    output_root = args.output
    save_empty = args.save_empty

    print("=" * 70)
    print("  NIfTI → 2D mask extraction (axial / coronal / sagittal)")
    print("=" * 70)

    # ── Load NIfTI ──
    if not os.path.isfile(nifti_path):
        print(f"  [ERROR] NIfTI file not found: {nifti_path}")
        sys.exit(1)

    print(f"\n[1] Loading NIfTI …")
    print(f"    {nifti_path}")
    nii = nib.load(nifti_path)
    vol = nii.get_fdata()
    print(f"    Shape:       {vol.shape}")
    print(f"    Orientation: {nib.aff2axcodes(nii.affine)}")
    print(f"    Voxel range: [{vol.min()}, {vol.max()}]")
    print(f"    Non-zero:    {np.count_nonzero(vol):,} voxels")

    # ── Binarise (in case values are floats) ──
    vol = (vol > 0).astype(np.uint8)

    # ── Define views ──
    views = [
        {
            "name": "axial",
            "label": "Axial (top view)",
            "axis": 2,
            "reorient": reorient_axial,
        },
        {
            "name": "coronal",
            "label": "Coronal (front view)",
            "axis": 1,
            "reorient": reorient_coronal,
        },
        {
            "name": "sagittal",
            "label": "Sagittal (side view)",
            "axis": 0,
            "reorient": reorient_sagittal,
        },
    ]

    # ── Extract slices ──
    print(f"\n[2] Extracting slices …")
    print(f"    Output root: {output_root}")
    print(f"    Save empty:  {save_empty}")
    t0 = time.time()

    results = []
    for v in views:
        out_dir = os.path.join(output_root, v["name"])
        saved, nonempty, total = save_slices(
            volume=vol,
            axis=v["axis"],
            axis_name=v["name"],
            output_dir=out_dir,
            reorient_fn=v["reorient"],
            save_empty=save_empty,
        )
        results.append((v["label"], saved, nonempty, total, out_dir))
        print(
            f"    ✅ {v['label']:25s}  {saved:4d} saved / {nonempty:4d} non-empty / {total:4d} total"
        )

    elapsed = time.time() - t0

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Extraction complete in {elapsed:.1f}s")
    print(f"")
    for label, saved, nonempty, total, out_dir in results:
        print(f"  {label}:")
        print(f"    Saved: {saved} images  |  Non-empty: {nonempty}/{total}")
        print(f"    Dir:   {out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
