# -*- coding: utf-8 -*-
"""
bbox_from_masks.py — Generate YOLO-format bounding boxes from segmentation
mask PNGs using connected-component (island) detection.

    Algorithm (per mask image):
    1. Binarise the mask (threshold > 0)
    2. Label connected components (8-connectivity) — each connected region
        is a separate "island" (bone)
    3. For each island, compute the bounding box from its pixel extent
    4. Add configurable padding (in pixels) around the tight bbox
    5. Clamp to image bounds
    6. Convert to YOLO format: <class> <x_center> <y_center> <width> <height>
        where all values are normalised to [0, 1] relative to image size
    7. Save as .txt alongside (or in a labels/ folder)

Optionally filters out tiny components (noise) via MIN_AREA.

Usage:
    python bbox_from_masks.py
    python bbox_from_masks.py --masks extracted_masks/axial --output labels/axial
    python bbox_from_masks.py --padding 5 --min-area 50
    python bbox_from_masks.py --view all
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import label as scipy_label
from natsort import natsorted

# =======================================================================
# ========================= DEFAULT CONFIG ==============================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"
BASE_DIR = os.path.join(PROJECT_ROOT, "pipeline_medsam2_3d")

# ── Input: mask folder (with axial/, coronal/, sagittal/ subfolders) ──
DEFAULT_MASKS_ROOT = os.path.join(BASE_DIR, "p18_extracted_masks")

# ── Output: labels folder ──
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "p18_labels")

# ── Padding: extra pixels added to each side of the bounding box ──
DEFAULT_PADDING = 3

# ── Minimum connected-component area (pixels) to keep ──
# Components smaller than this are treated as noise and skipped
DEFAULT_MIN_AREA = 20

# ── YOLO class ID for all bones (single-class detection) ──
# Change this or extend to multi-class if you label bone types differently
DEFAULT_CLASS_ID = 0

# ── Views to process ──
ALL_VIEWS = ["axial", "coronal", "sagittal"]

# =======================================================================
# ========================= HELPERS =====================================
# =======================================================================


def find_islands(mask_binary: np.ndarray, min_area: int = 20):
    """
    Find connected components (islands) in a binary mask.

    Uses scipy.ndimage.label with 8-connectivity (structuring element of ones).

    Args:
        mask_binary: 2D bool/uint8 array
        min_area:    minimum pixel count to keep a component

    Returns:
        List of dicts: [{"label": int, "bbox": (r_min, c_min, r_max, c_max), "area": int}, ...]
        bbox values are tight (no padding), pixel coordinates, inclusive.
    """
    # 8-connectivity structuring element
    struct = np.ones((3, 3), dtype=int)
    labelled, n_components = scipy_label(mask_binary, structure=struct)

    islands = []
    for comp_id in range(1, n_components + 1):
        component_mask = labelled == comp_id
        area = int(component_mask.sum())

        if area < min_area:
            continue

        rows, cols = np.where(component_mask)
        r_min, r_max = int(rows.min()), int(rows.max())
        c_min, c_max = int(cols.min()), int(cols.max())

        islands.append(
            {
                "label": comp_id,
                "bbox": (r_min, c_min, r_max, c_max),
                "area": area,
            }
        )

    return islands


def bbox_to_yolo(
    r_min: int,
    c_min: int,
    r_max: int,
    c_max: int,
    img_h: int,
    img_w: int,
    padding: int = 0,
    class_id: int = 0,
) -> str:
    """
    Convert a pixel bounding box to YOLO format with padding.

    Padding is added to each side:
        padded_width  = (c_max - c_min + 1) + 2 * padding
        padded_height = (r_max - r_min + 1) + 2 * padding

    Output: "<class> <x_center> <y_center> <width> <height>"
    All coordinates normalised to [0, 1].
    """
    # Add padding and clamp to image bounds
    r_min_p = max(0, r_min - padding)
    c_min_p = max(0, c_min - padding)
    r_max_p = min(img_h - 1, r_max + padding)
    c_max_p = min(img_w - 1, c_max + padding)

    # YOLO format: x_center, y_center, width, height (normalised)
    # Note: YOLO x = column, y = row
    box_w = c_max_p - c_min_p + 1
    box_h = r_max_p - r_min_p + 1
    x_center = (c_min_p + c_max_p) / 2.0 / img_w
    y_center = (r_min_p + r_max_p) / 2.0 / img_h
    w_norm = box_w / img_w
    h_norm = box_h / img_h

    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def process_mask_file(
    mask_path: str,
    output_path: str,
    padding: int,
    min_area: int,
    class_id: int,
) -> int:
    """
    Process a single mask PNG → YOLO label .txt file.

    Returns the number of bounding boxes written.
    """
    img = np.array(Image.open(mask_path).convert("L"))
    mask_bin = img > 0
    img_h, img_w = img.shape

    if not np.any(mask_bin):
        return 0

    islands = find_islands(mask_bin, min_area=min_area)

    if not islands:
        return 0

    lines = []
    for island in islands:
        r_min, c_min, r_max, c_max = island["bbox"]
        yolo_line = bbox_to_yolo(
            r_min,
            c_min,
            r_max,
            c_max,
            img_h,
            img_w,
            padding=padding,
            class_id=class_id,
        )
        lines.append(yolo_line)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return len(lines)


# =======================================================================
# ========================= MAIN =======================================
# =======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO bounding boxes from segmentation mask PNGs"
    )
    parser.add_argument(
        "--masks",
        type=str,
        default=DEFAULT_MASKS_ROOT,
        help="Root dir with mask PNGs (with axial/, coronal/, sagittal/ subfolders)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output dir for YOLO label .txt files",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=DEFAULT_PADDING,
        help="Padding pixels added to each side of the bounding box",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help="Minimum connected-component area (pixels) to generate a bbox",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=DEFAULT_CLASS_ID,
        help="YOLO class ID for all bounding boxes",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="all",
        choices=["all", "axial", "coronal", "sagittal"],
        help="Which view(s) to process",
    )
    args = parser.parse_args()

    masks_root = args.masks
    output_root = args.output
    padding = args.padding
    min_area = args.min_area
    class_id = args.class_id
    views = ALL_VIEWS if args.view == "all" else [args.view]

    print("=" * 70)
    print("  Bounding box generation from masks (connected-component islands)")
    print("=" * 70)
    print(f"  Masks root: {masks_root}")
    print(f"  Output:     {output_root}")
    print(f"  Padding:    {padding} px per side")
    print(f"  Min area:   {min_area} px")
    print(f"  Class ID:   {class_id}")
    print(f"  Views:      {', '.join(views)}")

    if not os.path.isdir(masks_root):
        print(f"\n  [ERROR] Masks root not found: {masks_root}")
        sys.exit(1)

    t0 = time.time()
    grand_total_files = 0
    grand_total_boxes = 0

    for view in views:
        mask_dir = os.path.join(masks_root, view)
        out_dir = os.path.join(output_root, view)

        if not os.path.isdir(mask_dir):
            print(f"\n  [WARN] Mask dir not found, skipping: {mask_dir}")
            continue

        mask_files = natsorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

        print(f"\n  [{view}] Processing {len(mask_files)} masks …")

        view_files = 0
        view_boxes = 0

        for mf in mask_files:
            mask_path = os.path.join(mask_dir, mf)
            label_name = os.path.splitext(mf)[0] + ".txt"
            label_path = os.path.join(out_dir, label_name)

            n_boxes = process_mask_file(
                mask_path=mask_path,
                output_path=label_path,
                padding=padding,
                min_area=min_area,
                class_id=class_id,
            )

            if n_boxes > 0:
                view_files += 1
                view_boxes += n_boxes

        print(f"    ✅ {view_files} files with {view_boxes} bounding boxes → {out_dir}")
        grand_total_files += view_files
        grand_total_boxes += view_boxes

    elapsed = time.time() - t0

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.1f}s")
    print(
        f"  Total: {grand_total_files} label files, {grand_total_boxes} bounding boxes"
    )
    print(f"  Padding: {padding} px per side")
    for view in views:
        out_dir = os.path.join(output_root, view)
        if os.path.isdir(out_dir):
            n = len([f for f in os.listdir(out_dir) if f.endswith(".txt")])
            print(f"    {view}/  →  {n} label files")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
