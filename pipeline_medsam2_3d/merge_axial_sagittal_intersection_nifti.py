# -*- coding: utf-8 -*-
"""
merge_axial_sagittal_intersection_nifti.py

Create a single NIfTI mask by keeping only the voxel-wise intersection
between axial and sagittal mask volumes from the same run config.

Usage:
    python merge_axial_sagittal_intersection_nifti.py
"""

import os
import sys
import numpy as np
import nibabel as nib

# ======================================================================
# ============================ CONFIG ===================================
# ======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"
RUN_SUFFIX = "N10_conf05_iou05"

AXIAL_NIFTI = os.path.join(
    PROJECT_ROOT,
    f"pipeline_medsam2_3d/p15_3d_axial_segmentation_{RUN_SUFFIX}.nii.gz",
)

SAGITTAL_NIFTI = os.path.join(
    PROJECT_ROOT,
    f"pipeline_medsam2_3d/p15_3d_sagittal_segmentation_{RUN_SUFFIX}.nii.gz",
)

OUTPUT_NIFTI = os.path.join(
    PROJECT_ROOT,
    f"pipeline_medsam2_3d/p15_3d_intersection_axial_sagittal_{RUN_SUFFIX}.nii.gz",
)

# ======================================================================
# ============================== MAIN ===================================
# ======================================================================


def main() -> None:
    print("=" * 70)
    print("  Axial ∩ Sagittal mask intersection → NIfTI")
    print("=" * 70)

    for p in [AXIAL_NIFTI, SAGITTAL_NIFTI]:
        if not os.path.isfile(p):
            print(f"[ERROR] Missing input NIfTI: {p}")
            sys.exit(1)

    axial_img = nib.load(AXIAL_NIFTI)
    sagittal_img = nib.load(SAGITTAL_NIFTI)

    if axial_img.shape != sagittal_img.shape:
        print("[ERROR] Shape mismatch:")
        print(f"  Axial:    {axial_img.shape}")
        print(f"  Sagittal: {sagittal_img.shape}")
        sys.exit(1)

    if not np.allclose(axial_img.affine, sagittal_img.affine):
        print("[ERROR] Affine mismatch between axial and sagittal NIfTI.")
        sys.exit(1)

    axial = axial_img.get_fdata() > 0
    sagittal = sagittal_img.get_fdata() > 0

    intersection = np.logical_and(axial, sagittal).astype(np.uint8)

    out_header = axial_img.header.copy()
    out_header.set_data_dtype(np.uint8)
    out_img = nib.Nifti1Image(intersection, axial_img.affine, out_header)

    nib.save(out_img, OUTPUT_NIFTI)

    axial_nz = int(np.count_nonzero(axial))
    sagittal_nz = int(np.count_nonzero(sagittal))
    inter_nz = int(np.count_nonzero(intersection))

    print(f"Axial voxels:      {axial_nz}")
    print(f"Sagittal voxels:   {sagittal_nz}")
    print(f"Intersection voxels: {inter_nz}")
    print(f"Saved: {OUTPUT_NIFTI}")
    print("Done.")


if __name__ == "__main__":
    main()
