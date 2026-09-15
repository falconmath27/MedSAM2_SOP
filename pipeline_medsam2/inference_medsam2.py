# -*- coding: utf-8 -*-
"""
inference_medsam2.py — Full YOLO → MedSAM2 pipeline for knee CT bone segmentation.

Drop-in replacement for the original `inference.py` that called MedSAM.
This version calls `MedSAM2_Inference.py` instead.

Edit the USER CONFIG section below before running.
"""

import os
import sys
import glob
import math
from subprocess import run
from PIL import Image
import numpy as np
from skimage import io
from ultralytics import YOLO

# =======================================================================
# ========================= USER CONFIG =================================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"
test_folder = os.path.join(PROJECT_ROOT, "yolo_sam/yolo_sam/converted_pngs_p15_ordered_z/")
output_folder = os.path.join(
    PROJECT_ROOT, "pipeline_medsam2/segmented_ct_slices_p15_full_medsam2_ordered_z/"
)

# ── MedSAM2 paths ──
medsam2_script = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "MedSAM2_Inference.py"
)
medsam2_repo = os.path.join(PROJECT_ROOT, "pipeline_medsam2/MedSAM2")
medsam2_checkpoint = "MedSAM2_latest.pt"  # inside repo/checkpoints/
medsam2_device = "cuda"  # GPU available (RTX 4060)

# ── YOLO paths ──
yolo_weights_path = os.path.join(
    PROJECT_ROOT,
    "yolo_sam/yolo_sam/For training/runs_yolov12/exp_colab/weights/best.pt",
)

# ── Detection tuning ──
BBOX_EXPANSION_RATIO = 0.05  # Expand bounding box by 5%
MERGE_DISTANCE_THRESHOLD = 2  # Merge boxes closer than 2 px
CONFIDENCE_THRESHOLD = 0.15  # Discard boxes below this confidence

# =======================================================================


# ── Output sub-folders ──
left_knee_folder = os.path.join(output_folder, "left_knee")
right_knee_folder = os.path.join(output_folder, "right_knee")
merged_folder = os.path.join(output_folder, "Merged")

os.makedirs(left_knee_folder, exist_ok=True)
os.makedirs(right_knee_folder, exist_ok=True)
os.makedirs(merged_folder, exist_ok=True)

# ── Load YOLO ──
model = YOLO(yolo_weights_path)


# ───────────────────────── Utility functions ───────────────────────────


def are_boxes_close(boxA, boxB, threshold=3):
    """True if centres of boxA and boxB are within *threshold* pixels."""
    _, x1A, y1A, x2A, y2A = boxA
    _, x1B, y1B, x2B, y2B = boxB
    cA = ((x1A + x2A) / 2, (y1A + y2A) / 2)
    cB = ((x1B + x2B) / 2, (y1B + y2B) / 2)
    return math.hypot(cA[0] - cB[0], cA[1] - cB[1]) < threshold


def filter_boxes_by_confidence(boxes, threshold=3):
    """Keep highest-confidence box when centres overlap."""
    boxes_sorted = sorted(boxes, key=lambda b: b[0], reverse=True)
    final = []
    for box in boxes_sorted:
        if not any(are_boxes_close(box, kept, threshold) for kept in final):
            final.append(box)
    return final


def expand_box(x1, y1, x2, y2, img_w, img_h, ratio=0.1):
    """Expand box by *ratio* and clip to image bounds."""
    w, h = x2 - x1, y2 - y1
    dx, dy = int((w * ratio) / 2), int((h * ratio) / 2)
    return [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(img_w - 1, x2 + dx),
        min(img_h - 1, y2 + dy),
    ]


# ───────────────────────── Main loop ──────────────────────────────────

image_paths = sorted(glob.glob(os.path.join(test_folder, "*.png")))

for image_path in image_paths:
    print(f"\n Processing: {image_path}")
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # ── YOLO detection ──
    results = model.predict(image_path, conf=0.1)

    left_knee_boxes = []
    right_knee_boxes = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x_mid = (x1 + x2) / 2.0
        if x_mid < img_width / 2.0:
            left_knee_boxes.append((conf, x1, y1, x2, y2))
            print(f"  Left:  ({x1},{y1})-({x2},{y2}) conf={conf:.2f}")
        else:
            right_knee_boxes.append((conf, x1, y1, x2, y2))
            print(f"  Right: ({x1},{y1})-({x2},{y2}) conf={conf:.2f}")

    left_knee_boxes = filter_boxes_by_confidence(
        left_knee_boxes, threshold=MERGE_DISTANCE_THRESHOLD
    )
    right_knee_boxes = filter_boxes_by_confidence(
        right_knee_boxes, threshold=MERGE_DISTANCE_THRESHOLD
    )

    # ── Run MedSAM2 for each knee ──
    def process_knee(knee_boxes, knee_folder):
        if not knee_boxes:
            return
        expanded = [
            expand_box(
                x1, y1, x2, y2, img_width, img_height, ratio=BBOX_EXPANSION_RATIO
            )
            for _, x1, y1, x2, y2 in knee_boxes
        ]

        out_path = os.path.join(knee_folder, image_name)
        os.makedirs(out_path, exist_ok=True)

        # Use sys.executable so we stay in the same venv
        run(
            [
                sys.executable,
                medsam2_script,
                "-i",
                image_path,
                "-o",
                out_path,
                "--box",
                str(expanded),
                "--medsam2_repo",
                medsam2_repo,
                "--checkpoint",
                medsam2_checkpoint,
                "--device",
                medsam2_device,
            ]
        )

    process_knee(left_knee_boxes, left_knee_folder)
    process_knee(right_knee_boxes, right_knee_folder)

    # ── Merge left + right seg masks ──
    left_seg_path = os.path.join(left_knee_folder, image_name)
    right_seg_path = os.path.join(right_knee_folder, image_name)

    left_files = (
        glob.glob(os.path.join(left_seg_path, "seg_*.png"))
        if os.path.exists(left_seg_path)
        else []
    )
    right_files = (
        glob.glob(os.path.join(right_seg_path, "seg_*.png"))
        if os.path.exists(right_seg_path)
        else []
    )
    all_seg_files = left_files + right_files

    if not all_seg_files:
        print(f"  No segmentations for {image_name}")
        continue

    merged_mask = None
    for seg_file in all_seg_files:
        seg_img = io.imread(seg_file)
        seg_bin = (seg_img > 0).astype(np.uint8)
        merged_mask = (
            seg_bin if merged_mask is None else np.maximum(merged_mask, seg_bin)
        )

    merged_out = os.path.join(merged_folder, f"seg_{image_name}.png")
    io.imsave(merged_out, merged_mask.astype(np.uint8) * 255, check_contrast=False)
    print(f"  Merged → {merged_out}")

print("\n✅ All images processed with MedSAM2. Results in:", output_folder)
