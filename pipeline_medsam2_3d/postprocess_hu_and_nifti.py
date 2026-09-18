# -*- coding: utf-8 -*-
"""
postprocess_hu_and_nifti.py — HU refinement + NIfTI reconstruction for
the 3D MedSAM2 pipeline.

This version correctly implements the 2-mask HU approach:
  - **Guardrail mask** = full SAM output (outer boundary, never grow beyond)
  - **Seed mask** = high-intensity cortical bone pixels extracted from the
    DICOM HU values *within* the SAM mask (≥ 88th percentile brightness)
  - Seeds are grown via morphological reconstruction by dilation through
    all HU-passing pixels (90–1500 HU) that lie inside the SAM guardrail.

This fills trabecular (spongy) bone that SAM may have missed, while the
guardrail prevents leakage into soft tissue.

Pipeline:
  1. Build index→DICOM mapping from dicom_metadata CSV
  2. For each merged seg PNG:
     a. Load SAM mask (guardrail)
     b. Load matching DICOM → HU array
     c. Extract high-intensity seed pixels from SAM mask region
     d. Grow seeds within (HU band ∩ SAM mask) via reconstruction
     e. Post-process (fill holes, morphological closing)
     f. Save refined PNG
  3. Stack all slices into 3D volume
  4. Save as NIfTI with correct affine from reference volume
"""

import os
import re
import sys
import time

import numpy as np
import pydicom
import nibabel as nib
from natsort import natsorted
from skimage import io
from skimage.morphology import binary_erosion, closing, disk
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, binary_dilation
from skimage.morphology import reconstruction

from dicom_metadata import scan_dicom_dir

# =======================================================================
# ========================= USER CONFIG =================================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"

# ── Input: merged segmentation masks from 3D MedSAM2 ──
merged_seg_folder = os.path.join(
    PROJECT_ROOT, "pipeline_medsam2_3d/segmented_ct_slices_p15_3d_v3/Merged"
)

# ── DICOM directory ──
dicom_dir = os.path.join(
    PROJECT_ROOT,
    "file 15 Dnyaneshwar Dhadam/file 15 Dnyaneshwar Dhadam/DICOM/20240520/16250000",
)

# ── Reference NIfTI (for affine + header) ──
reference_nifti = os.path.join(PROJECT_ROOT, "Pat15segbonecomplete.nii")

# ── Output ──
output_dir = os.path.join(PROJECT_ROOT, "pipeline_medsam2_3d/postprocessed_p15_3d")
output_hu_slices = os.path.join(output_dir, "hu_refined_slices")
os.makedirs(output_hu_slices, exist_ok=True)

# ── HU parameters ──
HU_GROW_MIN = 90
HU_GROW_MAX = 1500

# ── Seed extraction ──
# Within the SAM mask, pixels with HU above this percentile become seeds.
# 88th percentile picks up the bright cortical bone shell.
SEED_HU_PERCENTILE = 88
# Additional small dilation of seeds to bridge tiny cortical gaps
SEED_DILATE_PX = 1

# ── Post-processing ──
FILL_HOLES = True
CLOSING_RADIUS = 2

# ── Volume ──
TOTAL_SLICES = 704

# =======================================================================
# ========================= SETUP ======================================
# =======================================================================

print("=" * 70)
print("  Post-processing: HU refinement + NIfTI reconstruction (3D pipeline)")
print("=" * 70)

# ── Build index → DICOM filename mapping ──
print("\n[Setup] Scanning DICOM directory for Z-ordering …")
entries = scan_dicom_dir(dicom_dir)
if not entries:
    print(f"  [ERROR] No valid DICOMs found in {dicom_dir}")
    sys.exit(1)

index_to_dicom = {}
for i, entry in enumerate(entries):
    idx = i + 1  # 1-based
    index_to_dicom[idx] = entry["filename"]

print(
    f"  Mapped {len(index_to_dicom)} slices  |  Z range: {entries[0]['Z']} → {entries[-1]['Z']}"
)


# =======================================================================
# ========================= HELPERS ====================================
# =======================================================================


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


def extract_cortical_seeds(sam_mask, hu, percentile=88):
    """
    Extract high-intensity cortical bone pixels from within the SAM mask.

    Within the SAM-masked region, pixels above the given percentile of HU
    values are selected as seeds.  This picks up the bright cortical bone
    shell, which will then be grown inward to fill trabecular bone.

    Returns a boolean seed mask.
    """
    hu_in_mask = hu[sam_mask]
    if len(hu_in_mask) == 0:
        return np.zeros_like(sam_mask, dtype=bool)

    threshold = np.percentile(hu_in_mask, percentile)
    seeds = sam_mask & (hu >= threshold)
    return seeds


def hu_constrained_grow(seed_mask, guardrail_mask, hu, hu_min, hu_max, dilate_px=1):
    """
    Grow seed mask within HU bone band, constrained to guardrail (SAM) boundary.

    Uses morphological reconstruction by dilation:
      - growth_mask = (HU in [hu_min, hu_max]) AND guardrail
      - seeds are slightly dilated, then intersected with growth_mask
      - reconstruction grows seeds through all connected growth_mask pixels

    This fills trabecular bone interiors while staying within the SAM boundary.
    """
    hu_band = (hu >= hu_min) & (hu <= hu_max)
    growth_mask = hu_band & guardrail_mask

    # Slight dilation of seeds to bridge tiny cortical gaps
    if dilate_px > 0:
        seed_mask = binary_dilation(seed_mask, structure=disk(dilate_px))

    # Enforce seed ≤ mask (required by reconstruction)
    seed_mask = seed_mask & growth_mask

    if not np.any(seed_mask):
        return np.zeros_like(seed_mask, dtype=bool)

    seeds_u8 = seed_mask.astype(np.uint8)
    mask_u8 = growth_mask.astype(np.uint8)
    grown = reconstruction(seeds_u8, mask_u8, method="dilation")
    return grown > 0.5


# =======================================================================
# ============= STEP 1: SCAN SEGMENTATION MASKS =======================
# =======================================================================

print("\n[Step 1] Scanning merged segmentation folder …")
if not os.path.isdir(merged_seg_folder):
    print(f"  [WARN] Merged folder not found: {merged_seg_folder}")
    print(f"  Run inference_medsam2_3d.py first!")
    sys.exit(1)

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

if seg_index_map:
    print(f"  Index range: {min(seg_index_map.keys())} → {max(seg_index_map.keys())}")
else:
    print("  No segmentation masks found!")
    sys.exit(1)


# =======================================================================
# ============= STEP 2: HU REFINEMENT =================================
# =======================================================================

print(f"\n[Step 2] Applying HU refinement (HU band [{HU_GROW_MIN}, {HU_GROW_MAX}]) …")
print(f"  Seed extraction: top {100 - SEED_HU_PERCENTILE}% brightness within SAM mask")
t0 = time.time()

raw_slices = []
hu_slices = []
refined_count = 0
seed_empty_count = 0

for idx in range(1, TOTAL_SLICES + 1):
    if idx not in seg_index_map:
        raw_slices.append(None)
        hu_slices.append(None)
        continue

    # Load SAM segmentation mask (this is the GUARDRAIL)
    seg_path = seg_index_map[idx]
    seg_img = io.imread(seg_path)
    if seg_img.ndim == 3:
        seg_img = seg_img[..., 0]
    sam_mask = seg_img > 0

    # Store raw (before HU)
    raw_slices.append(sam_mask.astype(np.uint8))

    # Load DICOM for HU values
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
        hu_slices.append(sam_mask.astype(np.uint8))
        continue

    # Extract SEED mask: high-intensity cortical pixels within SAM region
    seed_mask = extract_cortical_seeds(sam_mask, hu, SEED_HU_PERCENTILE)

    if not np.any(seed_mask):
        # No bright pixels found — fall back to using SAM mask intersected with HU band
        seed_empty_count += 1
        hu_band = (hu >= HU_GROW_MIN) & (hu <= HU_GROW_MAX)
        refined = sam_mask & hu_band
    else:
        # Grow seeds within HU band, bounded by SAM guardrail
        refined = hu_constrained_grow(
            seed_mask=seed_mask,
            guardrail_mask=sam_mask,
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

    # Final clamp to SAM boundary (safety)
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
if seed_empty_count > 0:
    print(
        f"  [INFO] {seed_empty_count} slices had no bright seeds → used HU band fallback"
    )


# =======================================================================
# ============= STEP 3: BUILD NIfTI VOLUMES ============================
# =======================================================================

print("\n[Step 3] Building NIfTI volumes …")

ref_nii = nib.load(reference_nifti)
ref_affine = ref_nii.affine
ref_header = ref_nii.header.copy()
expected_hw = (ref_nii.shape[0], ref_nii.shape[1])

print(f"  Reference shape: {ref_nii.shape}  affine diagonal: {np.diag(ref_affine)[:3]}")


def build_volume(slices_list, expected_hw, total_slices):
    """Stack 2D mask slices into a 3D volume (H, W, Z)."""
    vol = np.zeros((expected_hw[0], expected_hw[1], total_slices), dtype=np.uint8)
    for z, sl in enumerate(slices_list):
        if sl is not None:
            if sl.shape == expected_hw:
                vol[:, :, z] = sl
            elif sl.shape == expected_hw[::-1]:
                vol[:, :, z] = sl.T
            else:
                from PIL import Image as PILImage

                sl_resized = np.array(
                    PILImage.fromarray(sl).resize(expected_hw[::-1], PILImage.NEAREST)
                )
                vol[:, :, z] = sl_resized
    return vol


def save_nifti(volume, affine, header, out_path):
    """Save volume as NIfTI with given affine and header."""
    nii = nib.Nifti1Image(volume, affine=affine, header=header)
    nii.set_data_dtype(np.uint8)
    nii.header["bitpix"] = 8
    nii.header["dim"][0] = 3
    nii.header["dim"][3] = volume.shape[2]
    nib.save(nii, out_path)


# ── Raw volume (no HU refinement) ──
raw_vol = build_volume(raw_slices, expected_hw, TOTAL_SLICES)
raw_path = os.path.join(output_dir, "p15_medsam2_3d_segmentation.nii.gz")
save_nifti(raw_vol, ref_affine, ref_header, raw_path)
raw_nonzero = int(np.count_nonzero(raw_vol))
print(f"  ✅ Raw NIfTI saved: {raw_path}")
print(f"     Shape: {raw_vol.shape}  |  Non-zero voxels: {raw_nonzero}")

# ── HU-refined volume ──
hu_vol = build_volume(hu_slices, expected_hw, TOTAL_SLICES)
hu_path = os.path.join(output_dir, "p15_medsam2_3d_segmentation_hu.nii.gz")
save_nifti(hu_vol, ref_affine, ref_header, hu_path)
hu_nonzero = int(np.count_nonzero(hu_vol))
print(f"  ✅ HU-refined NIfTI saved: {hu_path}")
print(f"     Shape: {hu_vol.shape}  |  Non-zero voxels: {hu_nonzero}")

if raw_nonzero > 0:
    reduction = (1.0 - hu_nonzero / raw_nonzero) * 100
else:
    reduction = 0

# =======================================================================
# ========================= SUMMARY ====================================
# =======================================================================

print(f"\n{'=' * 70}")
print(f"  Post-processing complete!")
print(f"  Slices with segmentation: {refined_count} / {TOTAL_SLICES}")
print(
    f"  Raw voxels: {raw_nonzero:,}  →  HU-refined voxels: {hu_nonzero:,}  ({reduction:.1f}% reduction)"
)
print(f"")
print(f"  Outputs:")
print(f"    Raw NIfTI:     {raw_path}")
print(f"    HU NIfTI:      {hu_path}")
print(f"    HU slices dir: {output_hu_slices}")
print(f"{'=' * 70}")
