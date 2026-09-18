# -*- coding: utf-8 -*-
"""
preview_masks.py — Overlay segmentation masks on CT image slices to
generate preview images for all three views (axial, coronal, sagittal).

For each view, this script:
  1. Finds matching pairs of CT image PNGs and mask PNGs (by filename index)
  2. Renders the mask as a semi-transparent coloured overlay on the grayscale CT
  3. Saves the composite to a preview folder

Usage:
    python preview_masks.py
    python preview_masks.py --pngs extracted_pngs --masks extracted_masks --output preview
    python preview_masks.py --color 255 0 0 --alpha 0.4
"""

import argparse
import os
import re
import sys
import time

import numpy as np
from PIL import Image
from natsort import natsorted

# =======================================================================
# ========================= DEFAULT CONFIG ==============================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"
BASE_DIR = os.path.join(PROJECT_ROOT, "pipeline_medsam2_3d")

# ── Input directories (each must contain axial/, coronal/, sagittal/ subfolders) ──
DEFAULT_PNGS_DIR = os.path.join(BASE_DIR, "p18_extracted_pngs")
DEFAULT_MASKS_DIR = os.path.join(BASE_DIR, "p18_extracted_masks")

# ── Output ──
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "p18_preview_masks")
# ── Overlay style ──
DEFAULT_COLOR = (255, 50, 50)  # red overlay
DEFAULT_ALPHA = 0.45  # mask opacity (0 = invisible, 1 = opaque)
DEFAULT_EDGE_COLOR = (255, 255, 0)  # yellow contour
DRAW_CONTOUR = True  # draw mask contour for visibility
CONTOUR_WIDTH = 1  # contour thickness in pixels

# ── Views to process ──
VIEWS = ["axial", "coronal", "sagittal"]

# =======================================================================
# ========================= HELPERS =====================================
# =======================================================================


def extract_index(fname: str) -> int | None:
    """Extract the numeric index from filenames like axial_0350.png → 350."""
    m = re.search(r"_(\d+)\.png$", fname)
    return int(m.group(1)) if m else None


def get_contour_mask(mask: np.ndarray) -> np.ndarray:
    """Get the contour (boundary) pixels of a binary mask via erosion."""
    from scipy.ndimage import binary_erosion

    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=CONTOUR_WIDTH)
    return mask & ~eroded


def overlay_mask_on_image(
    ct_gray: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
    draw_contour: bool = True,
    edge_color: tuple[int, int, int] = (255, 255, 0),
) -> np.ndarray:
    """
    Overlay a binary mask on a grayscale CT image.

    Returns an RGB uint8 array.
    """
    # Convert grayscale to RGB
    if ct_gray.ndim == 2:
        rgb = np.stack([ct_gray, ct_gray, ct_gray], axis=-1).astype(np.float32)
    else:
        rgb = ct_gray.astype(np.float32)

    mask_bool = mask > 0

    if np.any(mask_bool):
        # Semi-transparent fill
        overlay = np.array(color, dtype=np.float32)
        for c in range(3):
            rgb[:, :, c] = np.where(
                mask_bool,
                rgb[:, :, c] * (1 - alpha) + overlay[c] * alpha,
                rgb[:, :, c],
            )

        # Contour
        if draw_contour:
            contour = get_contour_mask(mask_bool)
            if np.any(contour):
                edge = np.array(edge_color, dtype=np.float32)
                for c in range(3):
                    rgb[:, :, c] = np.where(contour, edge[c], rgb[:, :, c])

    return np.clip(rgb, 0, 255).astype(np.uint8)


def process_view(
    view_name: str,
    pngs_dir: str,
    masks_dir: str,
    output_dir: str,
    color: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int]:
    """Process a single view: match CT+mask pairs, overlay, save."""
    png_folder = os.path.join(pngs_dir, view_name)
    mask_folder = os.path.join(masks_dir, view_name)
    out_folder = os.path.join(output_dir, view_name)
    os.makedirs(out_folder, exist_ok=True)

    # Build index maps
    png_files = (
        {extract_index(f): f for f in os.listdir(png_folder) if f.endswith(".png")}
        if os.path.isdir(png_folder)
        else {}
    )

    mask_files = (
        {extract_index(f): f for f in os.listdir(mask_folder) if f.endswith(".png")}
        if os.path.isdir(mask_folder)
        else {}
    )

    # Find matching indices (must have a CT image; mask may be absent → just CT)
    all_indices = sorted(png_files.keys())
    matched = 0
    saved = 0

    for idx in all_indices:
        if idx is None:
            continue

        # Load CT image
        ct_path = os.path.join(png_folder, png_files[idx])
        ct_img = np.array(Image.open(ct_path).convert("L"))

        # Load mask if available
        if idx in mask_files:
            mask_path = os.path.join(mask_folder, mask_files[idx])
            mask_img = np.array(Image.open(mask_path).convert("L"))
            matched += 1

            # Resize mask if dimensions don't match CT
            if mask_img.shape != ct_img.shape:
                mask_img = np.array(
                    Image.fromarray(mask_img).resize(
                        (ct_img.shape[1], ct_img.shape[0]), Image.NEAREST
                    )
                )
        else:
            mask_img = np.zeros_like(ct_img)

        # Create overlay
        preview = overlay_mask_on_image(
            ct_gray=ct_img,
            mask=mask_img,
            color=color,
            alpha=alpha,
            draw_contour=DRAW_CONTOUR,
            edge_color=DEFAULT_EDGE_COLOR,
        )

        # Save
        out_name = f"preview_{view_name}_{idx:04d}.png"
        Image.fromarray(preview, mode="RGB").save(os.path.join(out_folder, out_name))
        saved += 1

    return saved, matched


# =======================================================================
# ========================= MAIN =======================================
# =======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Overlay segmentation masks on CT images for preview"
    )
    parser.add_argument(
        "--pngs",
        type=str,
        default=DEFAULT_PNGS_DIR,
        help="Root dir with CT PNGs (must have axial/, coronal/, sagittal/ subfolders)",
    )
    parser.add_argument(
        "--masks",
        type=str,
        default=DEFAULT_MASKS_DIR,
        help="Root dir with mask PNGs (must have axial/, coronal/, sagittal/ subfolders)",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT, help="Output root directory"
    )
    parser.add_argument(
        "--color",
        type=int,
        nargs=3,
        default=list(DEFAULT_COLOR),
        metavar=("R", "G", "B"),
        help="Overlay color as R G B (0–255)",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="Overlay opacity (0.0–1.0)"
    )
    args = parser.parse_args()

    pngs_dir = args.pngs
    masks_dir = args.masks
    output_dir = args.output
    color = tuple(args.color)
    alpha = args.alpha

    print("=" * 70)
    print("  Mask preview generator (CT + mask overlay)")
    print("=" * 70)
    print(f"  CT PNGs:  {pngs_dir}")
    print(f"  Masks:    {masks_dir}")
    print(f"  Output:   {output_dir}")
    print(f"  Color:    RGB{color}  alpha={alpha}")

    # ── Validate ──
    for d, label in [(pngs_dir, "CT PNGs"), (masks_dir, "Masks")]:
        if not os.path.isdir(d):
            print(f"  [ERROR] {label} dir not found: {d}")
            sys.exit(1)

    # ── Process each view ──
    print()
    t0 = time.time()
    total_saved = 0
    total_matched = 0

    for view in VIEWS:
        saved, matched = process_view(
            view_name=view,
            pngs_dir=pngs_dir,
            masks_dir=masks_dir,
            output_dir=output_dir,
            color=color,
            alpha=alpha,
        )
        total_saved += saved
        total_matched += matched
        print(f"  ✅ {view:10s}  {saved:4d} previews  ({matched} with masks)")

    elapsed = time.time() - t0

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Total: {total_saved} preview images ({total_matched} with mask overlays)")
    for view in VIEWS:
        print(f"    {view}/  →  {os.path.join(output_dir, view)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
