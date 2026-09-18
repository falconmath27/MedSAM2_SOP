# -*- coding: utf-8 -*-
"""
extract_pngs.py — Extract 2D CT image slices from DICOMs as PNGs in all
three orthogonal views: axial (top), coronal (front), sagittal (side).

Pipeline:
  1. Load all DICOMs in Z-order → build a 3D HU volume
  2. Apply CT windowing (bone window by default)
  3. Slice along each axis, reorient, and save as ordered PNGs

Usage:
    python extract_pngs.py
    python extract_pngs.py --output /path/to/output
    python extract_pngs.py --window-center 400 --window-width 1500
"""

import argparse
import os
import sys
import time

import numpy as np
import pydicom
from PIL import Image

from dicom_metadata import scan_dicom_dir

# =======================================================================
# ========================= DEFAULT CONFIG ==============================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"

# ── DICOM directory ──
DICOM_DIR = os.path.join(
    PROJECT_ROOT,
    "file 18 Balaji Abhyankar/DICOM/20240603/16100000",
)

# ── Default output directory ──
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "pipeline_medsam2_3d/p18_extracted_pngs")
# ── CT windowing (bone window) ──
DEFAULT_WINDOW_CENTER = 400
DEFAULT_WINDOW_WIDTH = 1500

# ── Image dimensions ──
IMG_ROWS = 512
IMG_COLS = 512

# =======================================================================
# ========================= HELPERS =====================================
# =======================================================================


def load_dicom_hu(dicom_path: str) -> np.ndarray:
    """Read a DICOM file and return the HU array (float32)."""
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply CT windowing and return 0–255 uint8 image."""
    lower = center - width / 2
    upper = center + width / 2
    img = np.clip(hu, lower, upper)
    img = ((img - lower) / (upper - lower) * 255.0).astype(np.uint8)
    return img


def orient_slice(img: np.ndarray) -> np.ndarray:
    """Apply the same orientation transform used in 2d_to_nifti.py when building
    the mask NIfTI volume (rot90 CCW + flipud).  We must apply this to DICOM
    slices too so both volumes share the same coordinate space."""
    img = np.rot90(img, k=1)  # 90° CCW
    img = np.flipud(img)
    return img


# ── Reorientation functions (same logic as extract_masks.py) ──


def reorient_axial(slice_2d: np.ndarray) -> np.ndarray:
    """Reorient an axial slice for display (90° CW + flip LR)."""
    img = np.rot90(slice_2d, k=3)
    img = np.fliplr(img)
    return img


def reorient_coronal(slice_2d: np.ndarray) -> np.ndarray:
    """Reorient a coronal slice: transpose so Z is vertical, flip for superior-up."""
    img = slice_2d.T
    img = np.flipud(img)
    return img


def reorient_sagittal(slice_2d: np.ndarray) -> np.ndarray:
    """Reorient a sagittal slice: transpose so Z is vertical, flip for superior-up."""
    img = slice_2d.T
    img = np.flipud(img)
    return img


def save_view_slices(
    volume_u8: np.ndarray,
    axis: int,
    axis_name: str,
    output_dir: str,
    reorient_fn,
):
    """Slice the windowed uint8 volume along an axis, reorient, and save PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    n_slices = volume_u8.shape[axis]
    saved = 0

    for i in range(n_slices):
        if axis == 0:
            sl = volume_u8[i, :, :]
        elif axis == 1:
            sl = volume_u8[:, i, :]
        else:
            sl = volume_u8[:, :, i]

        sl = reorient_fn(sl)

        fname = f"{axis_name}_{i:04d}.png"
        Image.fromarray(sl, mode="L").save(os.path.join(output_dir, fname))
        saved += 1

    return saved, n_slices


# =======================================================================
# ========================= MAIN =======================================
# =======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract Z-ordered CT slices from DICOMs as PNGs (axial / coronal / sagittal)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Root output directory (subfolders axial/, coronal/, sagittal/ created inside)",
    )
    parser.add_argument(
        "--window-center",
        type=float,
        default=DEFAULT_WINDOW_CENTER,
        help="CT window center (HU)",
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=DEFAULT_WINDOW_WIDTH,
        help="CT window width (HU). Set to 0 for min-max normalisation.",
    )
    args = parser.parse_args()

    output_root = args.output
    wc = args.window_center
    ww = args.window_width

    print("=" * 70)
    print("  DICOM → PNG extraction (axial / coronal / sagittal)")
    print("=" * 70)

    # ── Validate ──
    if not os.path.isdir(DICOM_DIR):
        print(f"  [ERROR] DICOM dir not found: {DICOM_DIR}")
        sys.exit(1)

    # ── Step 1: Build 3D HU volume ──
    print(f"\n[1] Scanning DICOM directory for Z-ordering …")
    entries = scan_dicom_dir(DICOM_DIR)
    n_slices = len(entries)
    if n_slices == 0:
        print("  [ERROR] No valid DICOMs found!")
        sys.exit(1)
    print(f"    {n_slices} slices, Z range: {entries[0]['Z']} → {entries[-1]['Z']}")

    print(f"\n[2] Building 3D HU volume …")
    t0 = time.time()
    hu_vol = np.zeros((IMG_ROWS, IMG_COLS, n_slices), dtype=np.float32)
    loaded = 0

    for z_idx, entry in enumerate(entries):
        dicom_path = os.path.join(DICOM_DIR, entry["filename"])

        if not os.path.isfile(dicom_path):
            continue

        hu = load_dicom_hu(dicom_path)
        if hu.shape == (IMG_ROWS, IMG_COLS):
            hu_vol[:, :, z_idx] = orient_slice(hu)
        loaded += 1

        if loaded % 100 == 0:
            print(f"    Loaded {loaded}/{n_slices} …", end="\r")

    print(f"    Loaded {loaded}/{n_slices} DICOMs in {time.time() - t0:.1f}s")

    # ── Step 3: Apply windowing ──
    print(f"\n[3] Applying CT windowing (center={wc}, width={ww}) …")
    vol_u8 = apply_window(hu_vol, wc, ww)
    del hu_vol  # free memory

    # ── Step 4: Slice in all 3 views ──
    print(f"\n[4] Extracting slices …")
    print(f"    Output root: {output_root}")

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

    t1 = time.time()
    results = []
    for v in views:
        out_dir = os.path.join(output_root, v["name"])
        saved, total = save_view_slices(
            volume_u8=vol_u8,
            axis=v["axis"],
            axis_name=v["name"],
            output_dir=out_dir,
            reorient_fn=v["reorient"],
        )
        results.append((v["label"], saved, total, out_dir))
        print(f"    ✅ {v['label']:25s}  {saved:4d} saved / {total:4d} total")

    elapsed = time.time() - t1

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(
        f"  Extraction complete in {elapsed:.1f}s  (volume build: {time.time() - t0:.1f}s total)"
    )
    for label, saved, total, out_dir in results:
        print(f"  {label}:")
        print(f"    {saved}/{total} slices → {out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
