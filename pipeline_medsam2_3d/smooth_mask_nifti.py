# -*- coding: utf-8 -*-
"""
Smooth a binary 3D mask NIfTI and save a new NIfTI with "_smooth" suffix.

Default pipeline:
1) Gaussian smoothing on the binary mask (float space)
2) Threshold back to binary
3) Optional hole fill and small component removal

Usage:
    python smooth_mask_nifti.py --input /path/to/mask.nii.gz
    python smooth_mask_nifti.py --input /path/to/mask.nii.gz --sigma 1.0 --threshold 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter


def default_output_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".nii.gz"):
        return input_path.with_name(name[:-7] + "_smooth.nii.gz")
    if name.endswith(".nii"):
        return input_path.with_name(name[:-4] + "_smooth.nii")
    return input_path.with_name(name + "_smooth.nii.gz")


def smooth_mask(
    mask: np.ndarray,
    sigma: float,
    threshold: float,
    fill_holes: bool,
) -> np.ndarray:
    # Work in float space for Gaussian smoothing, then binarize again.
    smoothed = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    out = smoothed >= threshold

    if fill_holes:
        out = binary_fill_holes(out)
    return np.asarray(out, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth a 3D mask NIfTI")
    parser.add_argument("--input", required=True, help="Input binary mask NIfTI")
    parser.add_argument(
        "--output",
        default=None,
        help="Output NIfTI path (default: input name + _smooth)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma in voxels (default: 1.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold after Gaussian blur (default: 0.5)",
    )
    parser.add_argument(
        "--no-fill-holes",
        action="store_true",
        help="Disable 3D hole filling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input NIfTI not found: {input_path}")

    output_path = Path(args.output) if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sigma < 0:
        raise ValueError("sigma must be >= 0")
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    img = nib.load(str(input_path))
    arr = img.get_fdata()
    mask = arr > 0

    out_mask = smooth_mask(
        mask=mask,
        sigma=float(args.sigma),
        threshold=float(args.threshold),
        fill_holes=not args.no_fill_holes,
    )

    header = img.header.copy()
    header.set_data_dtype(np.uint8)

    out_img = nib.Nifti1Image(out_mask, img.affine, header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(output_path))

    print("Input:", input_path)
    print("Output:", output_path)
    print("Shape:", out_mask.shape)
    print("Sigma:", args.sigma)
    print("Threshold:", args.threshold)
    print("Fill holes:", not args.no_fill_holes)
    print("Input nonzero:", int(np.count_nonzero(mask)))
    print("Output nonzero:", int(np.count_nonzero(out_mask)))


if __name__ == "__main__":
    main()
