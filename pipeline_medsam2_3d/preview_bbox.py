# -*- coding: utf-8 -*-
"""
preview_bbox.py — Overlay segmentation masks AND bounding boxes on CT image
slices to generate preview images for all three views.

For each view, this script:
  1. Loads the CT grayscale PNG
  2. Overlays the segmentation mask (semi-transparent red fill + yellow contour)
  3. Draws YOLO bounding boxes from the label .txt files (green rectangles)
  4. Saves the composite to preview_bbox/

Usage:
    python preview_bbox.py
    python preview_bbox.py --pngs extracted_pngs --masks extracted_masks --labels labels --output preview_bbox
    python preview_bbox.py --mask-color 255 50 50 --bbox-color 0 255 0 --alpha 0.35
"""

import argparse
import os
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from natsort import natsorted

# =======================================================================
# ========================= DEFAULT CONFIG ==============================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"
BASE_DIR = os.path.join(PROJECT_ROOT, "pipeline_medsam2_3d")

# ── Input directories (each with axial/, coronal/, sagittal/ subfolders) ──
DEFAULT_PNGS_DIR = os.path.join(BASE_DIR, "p18_extracted_pngs")
DEFAULT_MASKS_DIR = os.path.join(BASE_DIR, "p18_extracted_masks")
DEFAULT_LABELS_DIR = os.path.join(BASE_DIR, "p18_labels")

# ── Output ──
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "p18_preview_bbox")

# ── Mask overlay style ──
DEFAULT_MASK_COLOR = (255, 50, 50)  # red
DEFAULT_MASK_ALPHA = 0.35  # slightly more transparent than preview_masks
DEFAULT_EDGE_COLOR = (255, 255, 0)  # yellow contour
DRAW_CONTOUR = True
CONTOUR_WIDTH = 1

# ── Bounding box style ──
DEFAULT_BBOX_COLOR = (0, 255, 0)  # green
DEFAULT_BBOX_WIDTH = 2  # line thickness

# ── Views to process ──
VIEWS = ["axial", "coronal", "sagittal"]

# =======================================================================
# ========================= HELPERS =====================================
# =======================================================================


def extract_index(fname: str) -> int | None:
    """Extract the numeric index from filenames like axial_0350.png → 350."""
    m = re.search(r"_(\d+)\.\w+$", fname)
    return int(m.group(1)) if m else None


def get_contour_mask(mask: np.ndarray) -> np.ndarray:
    """Get the contour (boundary) pixels of a binary mask via erosion."""
    from scipy.ndimage import binary_erosion

    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=CONTOUR_WIDTH)
    return mask & ~eroded


def overlay_mask_on_image(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Overlay a binary mask as a semi-transparent fill + contour on an RGB image."""
    rgb = rgb.astype(np.float32)
    mask_bool = mask > 0

    if np.any(mask_bool):
        overlay = np.array(color, dtype=np.float32)
        for c in range(3):
            rgb[:, :, c] = np.where(
                mask_bool,
                rgb[:, :, c] * (1 - alpha) + overlay[c] * alpha,
                rgb[:, :, c],
            )

        if DRAW_CONTOUR:
            contour = get_contour_mask(mask_bool)
            if np.any(contour):
                edge = np.array(DEFAULT_EDGE_COLOR, dtype=np.float32)
                for c in range(3):
                    rgb[:, :, c] = np.where(contour, edge[c], rgb[:, :, c])

    return np.clip(rgb, 0, 255).astype(np.uint8)


def parse_yolo_labels(label_path: str, img_w: int, img_h: int):
    """
    Parse a YOLO label .txt file and convert to pixel bounding boxes.

    YOLO format: <class> <x_center> <y_center> <width> <height>  (normalised)
    Returns list of (class_id, x1, y1, x2, y2) in pixel coordinates.
    """
    boxes = []
    if not os.path.isfile(label_path):
        return boxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )

            # Convert normalised YOLO → pixel coords
            box_w = w * img_w
            box_h = h * img_h
            x1 = int(xc * img_w - box_w / 2)
            y1 = int(yc * img_h - box_h / 2)
            x2 = int(xc * img_w + box_w / 2)
            y2 = int(yc * img_h + box_h / 2)

            # Clamp
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w - 1, x2)
            y2 = min(img_h - 1, y2)

            boxes.append((cls, x1, y1, x2, y2))

    return boxes


def draw_bboxes(
    img: Image.Image,
    boxes: list,
    color: tuple[int, int, int],
    width: int,
) -> Image.Image:
    """Draw bounding boxes on a PIL Image."""
    draw = ImageDraw.Draw(img)
    for cls, x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img


def build_index_map(folder: str, ext: str) -> dict[int, str]:
    """Build {index: filename} mapping from a folder."""
    if not os.path.isdir(folder):
        return {}
    return {extract_index(f): f for f in os.listdir(folder) if f.endswith(ext)}


def process_view(
    view_name: str,
    pngs_dir: str,
    masks_dir: str,
    labels_dir: str,
    output_dir: str,
    mask_color: tuple[int, int, int],
    mask_alpha: float,
    bbox_color: tuple[int, int, int],
    bbox_width: int,
) -> tuple[int, int, int]:
    """Process a single view: overlay mask + draw bboxes, save."""
    png_folder = os.path.join(pngs_dir, view_name)
    mask_folder = os.path.join(masks_dir, view_name)
    label_folder = os.path.join(labels_dir, view_name)
    out_folder = os.path.join(output_dir, view_name)
    os.makedirs(out_folder, exist_ok=True)

    png_map = build_index_map(png_folder, ".png")
    mask_map = build_index_map(mask_folder, ".png")
    label_map = build_index_map(label_folder, ".txt")

    all_indices = sorted(idx for idx in png_map if idx is not None)
    saved = 0
    with_mask = 0
    with_bbox = 0

    for idx in all_indices:
        # Load CT image
        ct_path = os.path.join(png_folder, png_map[idx])
        ct_img = np.array(Image.open(ct_path).convert("L"))
        img_h, img_w = ct_img.shape

        # Convert to RGB
        rgb = np.stack([ct_img, ct_img, ct_img], axis=-1)

        # Overlay mask if available
        if idx in mask_map:
            mask_path = os.path.join(mask_folder, mask_map[idx])
            mask_img = np.array(Image.open(mask_path).convert("L"))
            if mask_img.shape != ct_img.shape:
                mask_img = np.array(
                    Image.fromarray(mask_img).resize((img_w, img_h), Image.NEAREST)
                )
            rgb = overlay_mask_on_image(rgb, mask_img, mask_color, mask_alpha)
            with_mask += 1

        # Convert to PIL for drawing bboxes
        pil_img = Image.fromarray(rgb, mode="RGB")

        # Draw bounding boxes if available
        if idx in label_map:
            label_path = os.path.join(label_folder, label_map[idx])
            boxes = parse_yolo_labels(label_path, img_w, img_h)
            if boxes:
                pil_img = draw_bboxes(pil_img, boxes, bbox_color, bbox_width)
                with_bbox += 1

        # Save
        out_name = f"preview_{view_name}_{idx:04d}.png"
        pil_img.save(os.path.join(out_folder, out_name))
        saved += 1

    return saved, with_mask, with_bbox


# =======================================================================
# ========================= MAIN =======================================
# =======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Preview CT + mask overlay + YOLO bounding boxes"
    )
    parser.add_argument(
        "--pngs",
        type=str,
        default=DEFAULT_PNGS_DIR,
        help="Root dir with CT PNGs (axial/, coronal/, sagittal/)",
    )
    parser.add_argument(
        "--masks",
        type=str,
        default=DEFAULT_MASKS_DIR,
        help="Root dir with mask PNGs (axial/, coronal/, sagittal/)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=DEFAULT_LABELS_DIR,
        help="Root dir with YOLO label .txt files (axial/, coronal/, sagittal/)",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT, help="Output root directory"
    )
    parser.add_argument(
        "--mask-color",
        type=int,
        nargs=3,
        default=list(DEFAULT_MASK_COLOR),
        metavar=("R", "G", "B"),
        help="Mask overlay color",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_MASK_ALPHA,
        help="Mask overlay opacity (0.0–1.0)",
    )
    parser.add_argument(
        "--bbox-color",
        type=int,
        nargs=3,
        default=list(DEFAULT_BBOX_COLOR),
        metavar=("R", "G", "B"),
        help="Bounding box color",
    )
    parser.add_argument(
        "--bbox-width",
        type=int,
        default=DEFAULT_BBOX_WIDTH,
        help="Bounding box line thickness",
    )
    args = parser.parse_args()

    pngs_dir = args.pngs
    masks_dir = args.masks
    labels_dir = args.labels
    output_dir = args.output
    mask_color = tuple(args.mask_color)
    mask_alpha = args.alpha
    bbox_color = tuple(args.bbox_color)
    bbox_width = args.bbox_width

    print("=" * 70)
    print("  Bbox preview generator (CT + mask + bounding boxes)")
    print("=" * 70)
    print(f"  CT PNGs:    {pngs_dir}")
    print(f"  Masks:      {masks_dir}")
    print(f"  Labels:     {labels_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Mask:       RGB{mask_color}  alpha={mask_alpha}")
    print(f"  Bbox:       RGB{bbox_color}  width={bbox_width}px")

    for d, label in [
        (pngs_dir, "CT PNGs"),
        (masks_dir, "Masks"),
        (labels_dir, "Labels"),
    ]:
        if not os.path.isdir(d):
            print(f"  [ERROR] {label} dir not found: {d}")
            sys.exit(1)

    print()
    t0 = time.time()
    total_saved = 0
    total_mask = 0
    total_bbox = 0

    for view in VIEWS:
        saved, w_mask, w_bbox = process_view(
            view_name=view,
            pngs_dir=pngs_dir,
            masks_dir=masks_dir,
            labels_dir=labels_dir,
            output_dir=output_dir,
            mask_color=mask_color,
            mask_alpha=mask_alpha,
            bbox_color=bbox_color,
            bbox_width=bbox_width,
        )
        total_saved += saved
        total_mask += w_mask
        total_bbox += w_bbox
        print(
            f"  ✅ {view:10s}  {saved:4d} previews  ({w_mask} with masks, {w_bbox} with bboxes)"
        )

    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.1f}s")
    print(
        f"  Total: {total_saved} previews ({total_mask} with masks, {total_bbox} with bboxes)"
    )
    for view in VIEWS:
        print(f"    {view}/  →  {os.path.join(output_dir, view)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
