# -*- coding: utf-8 -*-
"""
postprocess_hu_and_nifti.py — HU refinement + NIfTI reconstruction.

Takes the merged segmentation PNGs from the batched MedSAM2 pipeline,
applies HU-constrained refinement using the original DICOM slices,
then stacks everything into a NIfTI volume.

Pipeline:
  1. Build index→DICOM mapping from dicom_metadata CSV
  2. For each merged seg PNG:
     a. Load the SAM mask
     b. Load matching DICOM → HU array
     c. Constrain mask to HU bone window (grow seeds within HU band ∩ SAM mask)
     d. Post-process (fill holes, morphological closing)
     e. Save refined PNG
  3. Stack all 704 slices (including empty ones) into a 3D volume
  4. Save as NIfTI with the correct affine from the reference volume
"""

import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
from natsort import natsorted
from skimage import io
from skimage.morphology import binary_dilation, disk, closing
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from skimage.morphology import reconstruction

# =======================================================================
# ========================= USER CONFIG =================================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"

# ── Input: merged segmentation masks from batched MedSAM2 ──
merged_seg_folder = os.path.join(
    PROJECT_ROOT,
    "pipeline_medsam2/segmented_ct_slices_p15_v2/Merged",
)

# ── DICOM directory (flat folder with numeric filenames) ──
dicom_dir = os.path.join(
    PROJECT_ROOT,
    "file 15 Dnyaneshwar Dhadam/file 15 Dnyaneshwar Dhadam/DICOM/20240520/16250000",
)

# ── CSV with DICOM metadata (filename, Z, InstanceNumber, etc.) ──
dicom_csv = os.path.join(PROJECT_ROOT, "yolo_sam/yolo_sam/dicom_metadata_p15.csv")

# ── Reference NIfTI (for affine & header) ──
reference_nifti = os.path.join(PROJECT_ROOT, "Pat15segbonecomplete.nii")

# ── Output ──
output_dir = os.path.join(PROJECT_ROOT, "pipeline_medsam2/postprocessed_p15_medsam2")
output_hu_slices = os.path.join(output_dir, "hu_refined_slices")
output_nifti_raw = os.path.join(output_dir, "p15_medsam2_segmentation.nii.gz")
output_nifti_hu = os.path.join(output_dir, "p15_medsam2_segmentation_hu.nii.gz")

# ── HU thresholding parameters ──
HU_GROW_MIN = 90  # Lower bound for bone HU (includes trabecular bone)
HU_GROW_MAX = 1500  # Upper bound for bone HU
SEED_DILATE_PX = 1  # Slight dilation of seed before growing

# ── Post-processing ──
CLOSING_RADIUS = 2
FILL_HOLES = True

# ── Total slices (must match reference NIfTI Z dimension) ──
TOTAL_SLICES = 704

# =======================================================================

os.makedirs(output_hu_slices, exist_ok=True)

print("=" * 60)
print("  Post-processing: HU refinement + NIfTI reconstruction")
print("=" * 60)


# ─── Step 0: Build index → DICOM filename mapping ─────────────────
print("\n[Step 0] Building slice index → DICOM mapping from CSV …")
df = pd.read_csv(dicom_csv)
df_sorted = df.sort_values("Z", ascending=True).reset_index(drop=True)
# Index 0 in df_sorted → ordered_z file 0001_p15.png → Z=380.7
# The seg files are named seg_0013_p15.png where 0013 is the ordered_z index

index_to_dicom = {}
for i, row in df_sorted.iterrows():
    idx = i + 1  # 1-based
    index_to_dicom[idx] = str(int(row["filename"]))

print(
    f"  Mapped {len(index_to_dicom)} slices  |  Z range: {df_sorted.Z.min()} → {df_sorted.Z.max()}"
)


# ─── Helper: load DICOM as HU array ──────────────────────────────
def load_dicom_hu(dicom_filename):
    """Load a DICOM file and return HU array."""
    fpath = os.path.join(dicom_dir, dicom_filename)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"DICOM not found: {fpath}")
    ds = pydicom.dcmread(fpath)
    arr = ds.pixel_array.astype(np.int16)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def hu_constrained_grow(seed_mask, sam_mask, hu, hu_min, hu_max, dilate_px=1):
    """
    Grow seed mask within HU bone band, constrained to SAM boundary.
    Uses morphological reconstruction by dilation.
    """
    hu_band = (hu >= hu_min) & (hu <= hu_max)
    growth_mask = hu_band & sam_mask  # only grow within SAM + HU band

    # Slight dilation of seeds
    if dilate_px > 0:
        seed_mask = binary_dilation(seed_mask, footprint=disk(dilate_px))

    # Enforce seed ≤ mask (required by reconstruction)
    seed_mask = seed_mask & growth_mask

    if not np.any(seed_mask):
        return np.zeros_like(seed_mask, dtype=bool)

    seeds_u8 = seed_mask.astype(np.uint8)
    mask_u8 = growth_mask.astype(np.uint8)
    grown = reconstruction(seeds_u8, mask_u8, method="dilation")
    return grown > 0.5


# ─── Step 1: Identify which slices have segmentation masks ─────────
print("\n[Step 1] Scanning merged segmentation folder …")
seg_files = [
    f
    for f in os.listdir(merged_seg_folder)
    if f.startswith("seg_") and f.endswith(".png")
]
seg_files = natsorted(seg_files)
print(f"  Found {len(seg_files)} merged segmentation masks")


# Parse index from filename: seg_0013_p15.png → 13
def parse_index(fname):
    m = re.search(r"seg_(\d+)", fname)
    return int(m.group(1)) if m else None


seg_index_map = {}
for f in seg_files:
    idx = parse_index(f)
    if idx is not None:
        seg_index_map[idx] = os.path.join(merged_seg_folder, f)

print(f"  Index range: {min(seg_index_map.keys())} → {max(seg_index_map.keys())}")


# ─── Step 2: HU refinement ────────────────────────────────────────
print(f"\n[Step 2] Applying HU refinement (HU band [{HU_GROW_MIN}, {HU_GROW_MAX}]) …")
t0 = time.time()

raw_slices = []  # For NIfTI without HU
hu_slices = []  # For NIfTI with HU
refined_count = 0

for idx in range(1, TOTAL_SLICES + 1):
    if idx not in seg_index_map:
        # No segmentation for this slice → empty mask
        raw_slices.append(None)
        hu_slices.append(None)
        continue

    # Load SAM segmentation mask
    seg_path = seg_index_map[idx]
    seg_img = io.imread(seg_path)
    if seg_img.ndim == 3:
        seg_img = seg_img[..., 0]
    sam_mask = seg_img > 0

    # Store raw (before HU)
    raw_slices.append(sam_mask.astype(np.uint8))

    # Load DICOM for HU
    dicom_fname = index_to_dicom.get(idx)
    if dicom_fname is None:
        hu_slices.append(sam_mask.astype(np.uint8))
        continue

    try:
        hu = load_dicom_hu(dicom_fname)
    except Exception as e:
        print(f"  [WARN] idx={idx}: DICOM load failed ({e}), using raw mask")
        hu_slices.append(sam_mask.astype(np.uint8))
        continue

    if sam_mask.shape != hu.shape:
        # Shape mismatch — skip HU for this slice
        hu_slices.append(sam_mask.astype(np.uint8))
        continue

    # HU-constrained growth: use SAM mask as both seeds and boundary
    refined = hu_constrained_grow(
        seed_mask=sam_mask,
        sam_mask=sam_mask,
        hu=hu,
        hu_min=HU_GROW_MIN,
        hu_max=HU_GROW_MAX,
        dilate_px=SEED_DILATE_PX,
    )

    # Post-processing
    if FILL_HOLES and np.any(refined):
        refined = binary_fill_holes(refined)
    if CLOSING_RADIUS > 0 and np.any(refined):
        refined = closing(refined, footprint=disk(CLOSING_RADIUS))

    # Clamp to SAM boundary
    refined = refined & sam_mask

    hu_mask = refined.astype(np.uint8)
    hu_slices.append(hu_mask)
    refined_count += 1

    # Save refined slice
    out_name = f"seg_{idx:04d}_hu.png"
    io.imsave(
        os.path.join(output_hu_slices, out_name),
        hu_mask * 255,
        check_contrast=False,
    )

    if refined_count % 50 == 0:
        print(f"  Refined {refined_count} slices …", end="\r")

print(f"\n  HU refinement done: {refined_count} slices in {time.time() - t0:.1f}s")


# ─── Step 3: Build NIfTI volumes ─────────────────────────────────
print("\n[Step 3] Building NIfTI volumes …")

# Load reference for affine & header
ref_nii = nib.load(reference_nifti)
ref_affine = ref_nii.affine
ref_header = ref_nii.header.copy()
expected_hw = (ref_nii.shape[0], ref_nii.shape[1])  # (512, 512)

print(f"  Reference shape: {ref_nii.shape}  affine diagonal: {np.diag(ref_affine)[:3]}")


def build_volume(slice_list, expected_hw, total_slices):
    """Stack a list of 2D masks (or None for empty) into (H, W, Z) uint8 volume."""
    H, W = expected_hw
    volume = np.zeros((H, W, total_slices), dtype=np.uint8)
    for i, s in enumerate(slice_list):
        if s is not None:
            if s.shape == expected_hw:
                volume[:, :, i] = s
            else:
                # Resize if needed
                from PIL import Image

                pil = Image.fromarray(s * 255).resize((W, H), Image.NEAREST)
                volume[:, :, i] = (np.array(pil) > 127).astype(np.uint8)
    return volume


# Raw volume (SAM masks without HU)
vol_raw = build_volume(raw_slices, expected_hw, TOTAL_SLICES)
nii_raw = nib.Nifti1Image(vol_raw, affine=ref_affine, header=ref_header)
nii_raw.set_data_dtype(np.uint8)
nib.save(nii_raw, output_nifti_raw)
print(f"  ✅ Raw NIfTI saved: {output_nifti_raw}")
print(f"     Shape: {vol_raw.shape}  |  Non-zero voxels: {np.count_nonzero(vol_raw)}")

# HU-refined volume
vol_hu = build_volume(hu_slices, expected_hw, TOTAL_SLICES)
nii_hu = nib.Nifti1Image(vol_hu, affine=ref_affine, header=ref_header)
nii_hu.set_data_dtype(np.uint8)
nib.save(nii_hu, output_nifti_hu)
print(f"  ✅ HU-refined NIfTI saved: {output_nifti_hu}")
print(f"     Shape: {vol_hu.shape}  |  Non-zero voxels: {np.count_nonzero(vol_hu)}")


# ─── Summary ────────────────────────────────────────────────────
raw_nz = np.count_nonzero(vol_raw)
hu_nz = np.count_nonzero(vol_hu)
reduction = (1 - hu_nz / raw_nz) * 100 if raw_nz > 0 else 0

print(f"\n{'=' * 60}")
print(f"  Post-processing complete!")
print(f"  Slices with segmentation: {len(seg_index_map)} / {TOTAL_SLICES}")
print(f"  HU-refined slices: {refined_count}")
print(
    f"  Raw voxels: {raw_nz:,}  →  HU-refined voxels: {hu_nz:,}  ({reduction:.1f}% reduction)"
)
print(f"")
print(f"  Outputs:")
print(f"    Raw NIfTI:     {output_nifti_raw}")
print(f"    HU NIfTI:      {output_nifti_hu}")
print(f"    HU slices dir: {output_hu_slices}")
print(f"{'=' * 60}")
