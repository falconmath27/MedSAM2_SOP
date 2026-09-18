# -*- coding: utf-8 -*-
"""
2d_to_nifti.py — Convert a folder of 2D segmentation mask PNGs into a
single NIfTI volume.

Builds the NIfTI affine from DICOM headers (no ground-truth reference needed).
If a reference NIfTI is provided, it uses that affine instead.

Orientation fix:
    The PNG masks are stored in standard image orientation (row-0 = top).
    To align with the NIfTI coordinate system we apply per-slice transforms:
        1.  np.rot90(slice, k=ROT90_K)   — counter-clockwise rotation
        2.  np.flipud(slice)               — vertical mirror  (if FLIP_UD)
        3.  np.fliplr(slice)               — horizontal mirror (if FLIP_LR)
    Adjust ROT90_K, FLIP_UD, FLIP_LR below until the overlay matches the CT.

Usage:
    python 2d_to_nifti.py
"""

import os
import re
import sys
import time

import numpy as np
import nibabel as nib
import pydicom
from skimage import io
from natsort import natsorted

from dicom_metadata import scan_dicom_dir

# =======================================================================
# ========================= USER CONFIG =================================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"
PATIENT_ID = os.getenv("PATIENT_ID", "15").strip()
DIRECTION = os.getenv("DIRECTION", "sagittal").strip().lower()
RUN_SUFFIX = os.getenv("RUN_SUFFIX", "N10_conf05_iou05").strip()

if DIRECTION not in {"axial", "coronal", "sagittal"}:
    raise ValueError(
        f"Invalid DIRECTION='{DIRECTION}'. Use one of: axial, coronal, sagittal"
    )

# ── Input: folder containing seg_*.png masks ──
MASK_FOLDER = os.path.join(
    PROJECT_ROOT,
    f"pipeline_medsam2_3d/segmented_ct_slices_p{PATIENT_ID}_3d_{DIRECTION}_{RUN_SUFFIX}/Merged",
)

# ── DICOM directory (used to build affine if no reference NIfTI) ──
DICOM_DIR = os.getenv(
    "DICOM_DIR", os.path.join(PROJECT_ROOT, "pipeline_medsam2_3d")
).strip()

# ── Reference NIfTI (OPTIONAL — set to None to build affine from DICOM) ──
# We only use this for the affine matrix + header, NOT the segmentation data.
# Set to None to compute the affine from the DICOM headers instead.
_ref_candidates = [
    os.path.join(
        PROJECT_ROOT, f"pipeline_medsam2_3d/Pat{PATIENT_ID}segbonecomplete.nii"
    ),
    os.path.join(
        PROJECT_ROOT, f"pipeline_medsam2_3d/pat{PATIENT_ID}segbonecomplete.nii"
    ),
    os.path.join(PROJECT_ROOT, f"Pat{PATIENT_ID}segbonecomplete.nii"),
    os.path.join(PROJECT_ROOT, f"pat{PATIENT_ID}segbonecomplete.nii"),
    os.path.join(PROJECT_ROOT, f"Pat{PATIENT_ID}.nii"),
    os.path.join(PROJECT_ROOT, f"pat{PATIENT_ID}.nii"),
]
REFERENCE_NIFTI = next((p for p in _ref_candidates if os.path.isfile(p)), None)
USE_REFERENCE_NIFTI = os.getenv("USE_REFERENCE_NIFTI", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
if not USE_REFERENCE_NIFTI:
    REFERENCE_NIFTI = None

# ── Output ──
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    f"pipeline_medsam2_3d/p{PATIENT_ID}_3d_{DIRECTION}_segmentation_{RUN_SUFFIX}.nii.gz",
)

# ── Total slices in the stacked direction ──
# Set <= 0 to auto-use full depth from target/reference volume.
TOTAL_SLICES = 0

# ── Which axis the 2D masks are stacked along ──
# axial: 2, coronal: 1, sagittal: 0
STACK_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}[DIRECTION]

# ── Optional: reverse source mask stack order before inserting in volume ──
# Useful when extracted PNG index direction is opposite to target DICOM Z order.
REVERSE_STACK_ORDER = os.getenv("REVERSE_STACK_ORDER", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

# ── Image dimensions (rows, cols) — from DICOM ──
IMG_ROWS = 512
IMG_COLS = 512

# ── Orientation transforms (applied per-slice before stacking) ──
# Rotate counter-clockwise by 90° × ROT90_K  (0=none, 1=90°, 2=180°, 3=270°)
ROT90_K = 1
# Flip vertically (top ↔ bottom) after rotation
FLIP_UD = True
# Flip horizontally (left ↔ right) after rotation
FLIP_LR = False

# Additional correction for sagittal-stack conversion only.
# This legacy fix is needed only for some reference-NIfTI workflows.
# For DICOM-based affine/orientation, enabling it can introduce a coronal flip.
SAGITTAL_FIX_HORIZONTAL_FLIP = True

# =======================================================================
# ========================= HELPERS =====================================
# =======================================================================


def parse_index(fname: str) -> int | None:
    """Extract slice index from seg_*.png names (e.g. seg_0013_p15, seg_sagittal_0000)."""
    stem = os.path.splitext(fname)[0]
    if not stem.startswith("seg_"):
        return None
    m = re.search(r"(\d+)", stem[4:])
    return int(m.group(1)) if m else None


def orient_slice(img: np.ndarray) -> np.ndarray:
    """Apply the configured rotation + flip transforms to a 2D slice."""
    if ROT90_K:
        img = np.rot90(img, k=ROT90_K)
    if FLIP_UD:
        img = np.flipud(img)
    if FLIP_LR:
        img = np.fliplr(img)
    if STACK_AXIS == 0 and SAGITTAL_FIX_HORIZONTAL_FLIP and USE_REFERENCE_NIFTI:
        img = np.fliplr(img)
    return img


def build_affine_from_dicom(dicom_dir: str) -> np.ndarray:
    """
    Build a NIfTI-compatible affine matrix from DICOM headers.

    Scans the DICOM directory to get Z-sorted slice metadata, then reads
    the first slice for orientation + origin.

    The affine encodes:
      - Voxel spacing (PixelSpacing + slice spacing)
      - Image orientation (ImageOrientationPatient)
      - Volume origin (ImagePositionPatient of the first Z-sorted slice)
    """
    # Get Z-sorted metadata from DICOMs directly
    entries = scan_dicom_dir(dicom_dir)
    if not entries:
        raise RuntimeError(f"No valid DICOMs found in {dicom_dir}")

    # Read the FIRST sorted DICOM for orientation + origin
    first_fname = entries[0]["filename"]
    first_path = os.path.join(dicom_dir, first_fname)
    ds = pydicom.dcmread(first_path, stop_before_pixels=True)

    # Pixel spacing (row_spacing, col_spacing) in mm
    ps = [float(x) for x in ds.PixelSpacing]  # [row_sp, col_sp]
    row_sp, col_sp = ps[0], ps[1]

    # Image orientation: two 3D direction cosines (row_dir, col_dir)
    iop = [float(x) for x in ds.ImageOrientationPatient]
    row_dir = np.array(iop[0:3])  # direction along columns (x increases)
    col_dir = np.array(iop[3:6])  # direction along rows (y increases)

    # Slice direction = cross product of row and col directions
    slice_dir = np.cross(row_dir, col_dir)

    # Slice spacing: use actual Z difference between first two slices
    if len(entries) >= 2:
        z0 = entries[0]["Z"]
        z1 = entries[1]["Z"]
        slice_sp = abs(z1 - z0)
        if slice_sp == 0:
            slice_sp = float(getattr(ds, "SliceThickness", 1.0))
    else:
        slice_sp = float(getattr(ds, "SliceThickness", 1.0))

    # Origin = ImagePositionPatient of the first sorted slice
    ipp = [float(x) for x in ds.ImagePositionPatient]

    # Build 4×4 affine
    # NIfTI convention: affine maps (i, j, k) → (x, y, z) in mm
    # Column 0: row_dir * col_spacing  (i-direction maps along image columns)
    # Column 1: col_dir * row_spacing  (j-direction maps along image rows)
    # Column 2: slice_dir * slice_spacing
    # Column 3: origin
    affine = np.eye(4)
    affine[0:3, 0] = row_dir * col_sp
    affine[0:3, 1] = col_dir * row_sp
    affine[0:3, 2] = slice_dir * slice_sp
    affine[0:3, 3] = ipp

    return affine


# =======================================================================
# ========================= MAIN =======================================
# =======================================================================


def main():
    print("=" * 70)
    print("  2D masks → NIfTI converter")
    print("=" * 70)

    # ── Validate inputs ──
    if not os.path.isdir(MASK_FOLDER):
        print(f"  [ERROR] Mask folder not found: {MASK_FOLDER}")
        sys.exit(1)

    # ── Get affine + header ──
    if REFERENCE_NIFTI and os.path.isfile(REFERENCE_NIFTI):
        print(f"\n[1] Using REFERENCE NIfTI for affine …")
        print(f"    {REFERENCE_NIFTI}")
        ref_nii = nib.load(REFERENCE_NIFTI)
        affine = ref_nii.affine
        header = ref_nii.header.copy()
        target_shape = tuple(int(x) for x in ref_nii.shape[:3])
    else:
        print(f"\n[1] Building affine from DICOM headers …")
        print(f"    DICOM dir: {DICOM_DIR}")
        affine = build_affine_from_dicom(DICOM_DIR)
        dicom_entries = scan_dicom_dir(DICOM_DIR)
        auto_rows = int(dicom_entries[0]["Rows"]) if dicom_entries else IMG_ROWS
        auto_cols = int(dicom_entries[0]["Columns"]) if dicom_entries else IMG_COLS
        auto_slices = len(dicom_entries)
        header = nib.Nifti1Header()
        header.set_data_dtype(np.uint8)
        header["pixdim"][1:4] = [
            np.linalg.norm(affine[0:3, 0]),
            np.linalg.norm(affine[0:3, 1]),
            np.linalg.norm(affine[0:3, 2]),
        ]
        target_shape = (
            auto_rows,
            auto_cols,
            TOTAL_SLICES if TOTAL_SLICES and TOTAL_SLICES > 0 else auto_slices,
        )

    # Determine in-plane mask size based on stacking axis
    if STACK_AXIS == 0:
        expected_hw = (target_shape[1], target_shape[2])
    elif STACK_AXIS == 1:
        expected_hw = (target_shape[0], target_shape[2])
    elif STACK_AXIS == 2:
        expected_hw = (target_shape[0], target_shape[1])
    else:
        raise ValueError(f"Invalid STACK_AXIS={STACK_AXIS}; expected 0, 1, or 2")

    stack_len = target_shape[STACK_AXIS]
    if TOTAL_SLICES and TOTAL_SLICES > 0:
        if TOTAL_SLICES != stack_len:
            print(
                f"    [WARN] TOTAL_SLICES={TOTAL_SLICES} but target_shape[STACK_AXIS]={stack_len}; "
                f"using min={min(TOTAL_SLICES, stack_len)}"
            )
        n_stack = min(TOTAL_SLICES, stack_len)
    else:
        n_stack = stack_len

    print(f"    Affine diagonal: {np.diag(affine)[:3]}")
    print(f"    Orientation: {nib.aff2axcodes(affine)}")
    print(f"    Target volume shape: {target_shape}")
    print(f"    Expected mask plane size: {expected_hw}  (STACK_AXIS={STACK_AXIS})")
    print(f"    Reverse stack order: {REVERSE_STACK_ORDER}")

    # ── Scan mask folder ──
    print(f"\n[2] Scanning mask folder …")
    print(f"    {MASK_FOLDER}")
    all_files = [
        f
        for f in os.listdir(MASK_FOLDER)
        if f.endswith(".png") and f.startswith("seg_")
    ]
    all_files = natsorted(all_files)
    print(f"    Found {len(all_files)} mask files")

    if not all_files:
        print("  [ERROR] No seg_*.png files found!")
        sys.exit(1)

    # Build index → filepath mapping
    index_map: dict[int, str] = {}
    for f in all_files:
        idx = parse_index(f)
        if idx is not None:
            index_map[idx] = os.path.join(MASK_FOLDER, f)

    idx_min, idx_max = min(index_map), max(index_map)
    print(f"    Index range: {idx_min} → {idx_max}  ({len(index_map)} masks)")

    # Support both 1-based (1..N) and 0-based (0..N-1) index naming
    zero_based = idx_min == 0 and idx_max <= (n_stack - 1)
    print(f"    Indexing mode: {'0-based' if zero_based else '1-based'}")

    # ── Show transform config ──
    print(f"\n[3] Orientation transforms:")
    print(f"    Rotate CCW: {ROT90_K * 90}°")
    print(f"    Flip UD:    {FLIP_UD}")
    print(f"    Flip LR:    {FLIP_LR}")

    # ── Build volume ──
    print(f"\n[4] Building volume {target_shape} …")
    t0 = time.time()

    vol = np.zeros(target_shape, dtype=np.uint8)
    loaded = 0

    for s in range(n_stack):
        src_s = (n_stack - 1 - s) if REVERSE_STACK_ORDER else s
        idx = src_s if zero_based else (src_s + 1)
        if idx not in index_map:
            continue

        img = io.imread(index_map[idx])

        # Convert to single-channel if needed
        if img.ndim == 3:
            img = img[..., 0]

        # Binarise
        mask = (img > 0).astype(np.uint8)

        # Apply orientation fix
        mask = orient_slice(mask)

        # Fit into expected in-plane shape
        if mask.shape == expected_hw:
            mask_fit = mask
        elif mask.shape == expected_hw[::-1]:
            mask_fit = mask.T
        else:
            from PIL import Image as PILImage

            mask_fit = np.array(
                PILImage.fromarray(mask).resize(expected_hw[::-1], PILImage.NEAREST)
            )

        # Insert plane into volume along selected axis
        if STACK_AXIS == 0:
            vol[s, :, :] = mask_fit
        elif STACK_AXIS == 1:
            vol[:, s, :] = mask_fit
        else:
            vol[:, :, s] = mask_fit
        loaded += 1

    elapsed = time.time() - t0
    nz = int(np.count_nonzero(vol))
    print(f"    Loaded {loaded} masks, skipped {n_stack - loaded} empty slices")
    print(f"    Non-zero voxels: {nz:,}")
    print(f"    Time: {elapsed:.1f}s")

    # ── Save NIfTI ──
    print(f"\n[5] Saving NIfTI …")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    nii = nib.Nifti1Image(vol, affine=affine, header=header)
    nii.set_data_dtype(np.uint8)
    nii.header["bitpix"] = 8
    nii.header["dim"][0] = 3
    nii.header["dim"][3] = vol.shape[2]
    nib.save(nii, OUTPUT_PATH)

    print(f"    ✅ Saved: {OUTPUT_PATH}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Done!")
    print(f"  Volume shape:    {vol.shape}")
    print(f"  Non-zero voxels: {nz:,}")
    print(f"  Masks loaded:    {loaded} / {n_stack}")
    print(f"  Output:          {OUTPUT_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
