# -*- coding: utf-8 -*-
"""
bbox_from_masks_erosion.py — Generate YOLO-format bounding boxes from
segmentation mask PNGs, with erosion-based splitting for bones connected
by thin pixel bridges.

    Problem:
        Two separate bones sometimes touch or overlap by a thin bridge of
        just a few pixels in the mask.  Plain connected-component labelling
        sees them as one island and outputs one giant bounding box.

    Solution — erosion + watershed:
        1. Binarise the mask and find connected components (8-connectivity).
        2. For each component, erode it by `k` pixels (shrink the boundary).
           - If the eroded version stays as one piece → keep as one bbox.
           - If the eroded version splits into 2+ pieces → the bones were
             only connected by a thin bridge (≤ 2k pixels wide).
        3. When a split IS detected, use watershed to grow the eroded
           seeds back to reclaim all original pixels.  This assigns every
           pixel of the original component to its nearest eroded sub-piece.
        4. Compute bounding boxes from the ORIGINAL (full-size) pixels
           of each watershed region — so boxes are NOT shrunk.

    Visual example (erosion_radius = 2):

        Original mask           After erosion (k=2)        Watershed result
        ┌─────────────┐         ┌─────────────┐            ┌─────────────┐
        │ AAAAAAA.BBB │         │ ..AAA...B.. │            │ AAAAAAA BBB │
        │ AAAAAAA.BBB │         │ ..AAA...B.. │            │ AAAAAAA BBB │
        │ .AAAA##BB.. │   →     │ .........  │    →        │ .AAAA  BB.. │
        │ ..AAAA.BBB. │         │ ..AA....B. │            │ ..AAAA BBB. │
        │ ..AAAA.BBB. │         │ ..AA....B. │            │ ..AAAA BBB. │
        └─────────────┘         └─────────────┘            └─────────────┘
             1 bbox              2 eroded pieces             2 bboxes

        ## = thin bridge (≤ 4px wide, eroded away)
        Watershed grows A-seeds and B-seeds back into the original mask,
        splitting along the midline of the old bridge.

Usage:
    python bbox_from_masks_erosion.py
    python bbox_from_masks_erosion.py --erosion 3
    python bbox_from_masks_erosion.py --masks p4_extracted_masks --output p4_labels_erosion
    python bbox_from_masks_erosion.py --view axial --erosion 2 --padding 5
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import label as scipy_label
from scipy.ndimage import binary_erosion as scipy_binary_erosion
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from natsort import natsorted

# =======================================================================
# ========================= DEFAULT CONFIG ==============================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"
BASE_DIR = os.path.join(PROJECT_ROOT, "pipeline_medsam2_3d")

# ── Input: mask folder (with axial/, coronal/, sagittal/ subfolders) ──
DEFAULT_MASKS_ROOT = os.path.join(BASE_DIR, "p15_extracted_masks")

# ── Output: labels folder ──
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "p15_labels_erosion")

# ── Padding: extra pixels added to each side of the bounding box ──
DEFAULT_PADDING = 3

# ── Minimum connected-component area (pixels) to keep ──
DEFAULT_MIN_AREA = 20

# ── YOLO class ID for all bones (single-class detection) ──
DEFAULT_CLASS_ID = 0

# ── Erosion radius (pixels) for bridge splitting ──
# Bridges ≤ 2 * erosion_radius pixels wide will be split.
# Set to 0 to disable (behaves like the plain bbox_from_masks.py).
DEFAULT_EROSION = 2

# ── Views to process ──
ALL_VIEWS = ["axial", "coronal", "sagittal"]

# =======================================================================
# ========================= HELPERS =====================================
# =======================================================================

# 8-connectivity structuring element (shared)
STRUCT_8CONN = np.ones((3, 3), dtype=int)


def find_raw_islands(mask_binary: np.ndarray, min_area: int = 20):
    """
    Plain connected-component labelling (8-connectivity).

    Returns:
        labelled:  2D int array (0 = background, 1..N = component IDs)
        islands:   list of dicts with "label", "bbox", "area"
    """
    labelled, n = scipy_label(mask_binary.astype(np.uint8), structure=STRUCT_8CONN)

    islands = []
    for comp_id in range(1, n + 1):
        comp_mask = labelled == comp_id
        area = int(comp_mask.sum())

        if area < min_area:
            labelled[comp_mask] = 0  # zero out noise
            continue

        rows, cols = np.where(comp_mask)
        islands.append(
            {
                "label": comp_id,
                "bbox": (
                    int(rows.min()),
                    int(cols.min()),
                    int(rows.max()),
                    int(cols.max()),
                ),
                "area": area,
            }
        )

    return labelled, islands


def find_islands_with_erosion(
    mask_binary: np.ndarray,
    min_area: int = 20,
    erosion_radius: int = 2,
):
    """
    Connected-component labelling with erosion-based bridge splitting.

    Algorithm for each initial component:
      1. Erode by `erosion_radius` pixels using a square structuring element.
      2. Label the eroded mask.
         - If still 1 component → no bridge → keep original bbox.
         - If 0 components (erosion ate it all) → too small → keep original.
         - If 2+ components → bridge detected → proceed to step 3.
      3. Use the eroded sub-components as seeds in a watershed.
         The elevation map = negative Euclidean distance transform of the
         original component (so pixels near the centre are "low" = filled
         first).  Watershed grows each seed outward through the original
         mask until they meet, splitting along the natural midline.
      4. Compute bboxes from each watershed region's ORIGINAL pixels.

    Returns:
        List of dicts: [{"label": int, "bbox": (r_min, c_min, r_max, c_max),
                         "area": int}, ...]
    """
    # Step 1: plain labelling on original mask
    orig_labelled, orig_islands = find_raw_islands(mask_binary, min_area=min_area)

    if erosion_radius <= 0 or not orig_islands:
        return orig_islands, 0

    # Build the erosion structuring element (square kernel of side 2k+1)
    ek = 2 * erosion_radius + 1
    erosion_struct = np.ones((ek, ek), dtype=int)

    # Min area for eroded sub-pieces (allow them to be smaller since they're
    # shrunken; we still enforce full min_area on the final watershed regions)
    eroded_min_area = max(1, min_area // 4)

    final_islands = []
    next_id = 1
    split_count = 0

    for island in orig_islands:
        comp_mask = orig_labelled == island["label"]  # bool mask of this component

        # ── Erode ──
        eroded = scipy_binary_erosion(comp_mask, structure=erosion_struct)

        if not np.any(eroded):
            # Erosion consumed the entire component → keep as-is
            final_islands.append(
                {
                    "label": next_id,
                    "bbox": island["bbox"],
                    "area": island["area"],
                }
            )
            next_id += 1
            continue

        # ── Label eroded mask ──
        eroded_labelled, eroded_islands = find_raw_islands(
            eroded, min_area=eroded_min_area
        )

        if len(eroded_islands) <= 1:
            # No split → keep original component as one bbox
            final_islands.append(
                {
                    "label": next_id,
                    "bbox": island["bbox"],
                    "area": island["area"],
                }
            )
            next_id += 1
            continue

        # ── Split detected!  Watershed to reclaim eroded pixels ──
        split_count += 1

        # Build seed markers: relabel eroded sub-components as 1, 2, 3, …
        seeds = np.zeros_like(comp_mask, dtype=np.int32)
        for i, ei in enumerate(eroded_islands, start=1):
            seeds[eroded_labelled == ei["label"]] = i

        # Elevation = negative distance transform of original component.
        # Pixels far from the boundary are "deep" → filled first by watershed.
        # This makes the split line fall along the thinnest bridge.
        dist = distance_transform_edt(comp_mask)
        elevation = -dist

        # Watershed: grow seeds through original component mask only
        ws_labels = watershed(elevation, markers=seeds, mask=comp_mask)

        # ── Extract bboxes from each watershed region ──
        for sub_id in range(1, len(eroded_islands) + 1):
            sub_mask = ws_labels == sub_id
            sub_area = int(sub_mask.sum())

            if sub_area < min_area:
                continue

            rows, cols = np.where(sub_mask)
            final_islands.append(
                {
                    "label": next_id,
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                    "area": sub_area,
                }
            )
            next_id += 1

    return final_islands, split_count


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

    Output: "<class> <x_center> <y_center> <width> <height>"
    All coordinates normalised to [0, 1].
    """
    r_min_p = max(0, r_min - padding)
    c_min_p = max(0, c_min - padding)
    r_max_p = min(img_h - 1, r_max + padding)
    c_max_p = min(img_w - 1, c_max + padding)

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
    erosion_radius: int = 2,
) -> tuple[int, int]:
    """
    Process a single mask PNG → YOLO label .txt file.

    Returns (n_boxes, n_splits) — boxes written, components that were split.
    """
    img = np.array(Image.open(mask_path).convert("L"))
    mask_bin = img > 0
    img_h, img_w = img.shape

    if not np.any(mask_bin):
        return 0, 0

    result = find_islands_with_erosion(
        mask_bin,
        min_area=min_area,
        erosion_radius=erosion_radius,
    )

    if erosion_radius > 0:
        islands, n_splits = result
    else:
        islands, n_splits = result, 0

    if not islands:
        return 0, 0

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

    return len(lines), n_splits


# =======================================================================
# ========================= MAIN =======================================
# =======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO bboxes from masks with erosion-based bridge splitting"
    )
    parser.add_argument(
        "--masks",
        type=str,
        default=DEFAULT_MASKS_ROOT,
        help="Root dir with mask PNGs (axial/, coronal/, sagittal/ subfolders)",
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
        "--erosion",
        type=int,
        default=DEFAULT_EROSION,
        help="Erosion radius in pixels (bridges ≤ 2×erosion wide get split). 0 = disabled.",
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
    erosion_radius = args.erosion
    views = ALL_VIEWS if args.view == "all" else [args.view]

    print("=" * 70)
    print("  Bounding box generation (erosion-based bridge splitting)")
    print("=" * 70)
    print(f"  Masks root:     {masks_root}")
    print(f"  Output:         {output_root}")
    print(f"  Padding:        {padding} px per side")
    print(f"  Min area:       {min_area} px")
    print(
        f"  Erosion radius: {erosion_radius} px  (splits bridges ≤ {2 * erosion_radius} px wide)"
    )
    print(f"  Class ID:       {class_id}")
    print(f"  Views:          {', '.join(views)}")

    if not os.path.isdir(masks_root):
        print(f"\n  [ERROR] Masks root not found: {masks_root}")
        sys.exit(1)

    t0 = time.time()
    grand_total_files = 0
    grand_total_boxes = 0
    grand_total_splits = 0

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
        view_splits = 0

        for mf in mask_files:
            mask_path = os.path.join(mask_dir, mf)
            label_name = os.path.splitext(mf)[0] + ".txt"
            label_path = os.path.join(out_dir, label_name)

            n_boxes, n_splits = process_mask_file(
                mask_path=mask_path,
                output_path=label_path,
                padding=padding,
                min_area=min_area,
                class_id=class_id,
                erosion_radius=erosion_radius,
            )

            if n_boxes > 0:
                view_files += 1
                view_boxes += n_boxes
                view_splits += n_splits

        print(
            f"    ✅ {view_files} files, {view_boxes} boxes, {view_splits} erosion splits → {out_dir}"
        )
        grand_total_files += view_files
        grand_total_boxes += view_boxes
        grand_total_splits += view_splits

    elapsed = time.time() - t0

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.1f}s")
    print(
        f"  Total: {grand_total_files} label files, {grand_total_boxes} bounding boxes"
    )
    print(
        f"  Erosion splits: {grand_total_splits} components were split into multiple boxes"
    )
    print(
        f"  Settings: padding={padding}px, erosion={erosion_radius}px, min_area={min_area}px"
    )
    for view in views:
        out_dir = os.path.join(output_root, view)
        if os.path.isdir(out_dir):
            n = len([f for f in os.listdir(out_dir) if f.endswith(".txt")])
            print(f"    {view}/  →  {n} label files")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
