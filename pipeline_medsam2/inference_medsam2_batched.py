# -*- coding: utf-8 -*-
"""
inference_medsam2_batched.py — Batched YOLO → MedSAM2 pipeline.

Key differences from inference_medsam2.py:
  1. MedSAM2 model is loaded ONCE at startup (not per-image via subprocess).
  2. YOLO detection is batched (configurable batch size).
  3. All MedSAM2 inference happens in-process — no subprocess overhead.

Each image is still segmented individually in 2D mode (no 3D propagation),
but the elimination of ~2 model loads + subprocess spawns per image gives
a major speedup (~5-10x).
"""

import os
import sys
import glob
import math
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from ultralytics import YOLO
import torch

# =======================================================================
# =========================== CONFIG ====================================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"
test_folder = os.path.join(PROJECT_ROOT, "yolo_sam/yolo_sam/converted_pngs_p15_ordered_z/")
output_folder = os.path.join(
    PROJECT_ROOT, "pipeline_medsam2/segmented_ct_slices_p15_v2/"
)

# ── MedSAM2 paths ──
medsam2_repo = os.path.join(PROJECT_ROOT, "pipeline_medsam2/MedSAM2")
medsam2_checkpoint = "MedSAM2_latest.pt"  # inside repo/checkpoints/
medsam2_model_cfg = "configs/sam2.1_hiera_t512.yaml"
medsam2_device = "cuda"

# ── YOLO paths ──
yolo_weights_path = os.path.join(
    PROJECT_ROOT,
    "yolo_sam/yolo_sam/For training/runs_yolov12/exp_colab/weights/best.pt",
)

# ── Batching ──
YOLO_BATCH_SIZE = 32  # images per YOLO batch

# ── Detection tuning ──
BBOX_EXPANSION_RATIO = 0.05
MERGE_DISTANCE_THRESHOLD = 2
CONFIDENCE_THRESHOLD = 0.15

# ── Visualisation ──
SAVE_VIS = True  # set False to skip per-box overlay PNGs (saves time)

# =======================================================================
# ========================= SETUP ======================================
# =======================================================================

# ── Output sub-folders ──
left_knee_folder = os.path.join(output_folder, "left_knee")
right_knee_folder = os.path.join(output_folder, "right_knee")
merged_folder = os.path.join(output_folder, "Merged")
os.makedirs(left_knee_folder, exist_ok=True)
os.makedirs(right_knee_folder, exist_ok=True)
os.makedirs(merged_folder, exist_ok=True)


# ── Load MedSAM2 ONCE ─────────────────────────────────────────────
print("[INIT] Loading MedSAM2 model …")
t0 = time.time()

medsam2_root = os.path.abspath(medsam2_repo)
if medsam2_root not in sys.path:
    sys.path.insert(0, medsam2_root)

# Hydra needs CWD to be the repo root to resolve config paths
original_cwd = os.getcwd()
os.chdir(medsam2_root)

from sam2.build_sam import build_sam2_video_predictor_npz  # noqa: E402

ckpt_path = os.path.join(medsam2_root, "checkpoints", medsam2_checkpoint)
if not os.path.isfile(ckpt_path):
    sys.exit(f"[ERROR] Checkpoint not found: {ckpt_path}")

predictor = build_sam2_video_predictor_npz(
    medsam2_model_cfg, ckpt_path, device=medsam2_device
)
os.chdir(original_cwd)

print(f"[INIT] MedSAM2 loaded in {time.time() - t0:.1f}s")


# ── Load YOLO ONCE ────────────────────────────────────────────────
print("[INIT] Loading YOLO model …")
yolo_model = YOLO(yolo_weights_path)
print("[INIT] YOLO loaded ✓")


# =======================================================================
# ========================= HELPERS ====================================
# =======================================================================

IMAGE_SIZE = 512  # MedSAM2 internal resolution


def are_boxes_close(boxA, boxB, threshold=3):
    _, x1A, y1A, x2A, y2A = boxA
    _, x1B, y1B, x2B, y2B = boxB
    cA = ((x1A + x2A) / 2, (y1A + y2A) / 2)
    cB = ((x1B + x2B) / 2, (y1B + y2B) / 2)
    return math.hypot(cA[0] - cB[0], cA[1] - cB[1]) < threshold


def filter_boxes_by_confidence(boxes, threshold=3):
    boxes_sorted = sorted(boxes, key=lambda b: b[0], reverse=True)
    final = []
    for box in boxes_sorted:
        if not any(are_boxes_close(box, kept, threshold) for kept in final):
            final.append(box)
    return final


def expand_box(x1, y1, x2, y2, img_w, img_h, ratio=0.1):
    w, h = x2 - x1, y2 - y1
    dx, dy = int((w * ratio) / 2), int((h * ratio) / 2)
    return [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(img_w - 1, x2 + dx),
        min(img_h - 1, y2 + dy),
    ]


def prepare_image_tensor(img_3c: np.ndarray) -> torch.Tensor:
    """Resize H×W×3 uint8 → (1, 3, 512, 512) float, ImageNet-normalised."""
    pil = (
        Image.fromarray(img_3c)
        .convert("RGB")
        .resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    )
    arr = np.array(pil).astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).unsqueeze(0)


def medsam2_segment_box(img_tensor, box_np, H, W):
    """
    Run MedSAM2 on one image + one box, using the already-loaded predictor.
    Returns binary mask (H, W) uint8.
    """
    device = medsam2_device
    img_t = img_tensor.to(device)

    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.bfloat16, enabled=(device != "cpu")),
    ):
        state = predictor.init_state(img_t, H, W)

        scale_x = IMAGE_SIZE / W
        scale_y = IMAGE_SIZE / H
        box_scaled = np.array(
            [
                box_np[0] * scale_x,
                box_np[1] * scale_y,
                box_np[2] * scale_x,
                box_np[3] * scale_y,
            ]
        )

        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            box=box_scaled,
        )

        all_masks = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            all_masks[frame_idx] = {
                oid: (mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, oid in enumerate(obj_ids)
            }
        predictor.reset_state(state)

    if 0 in all_masks and 1 in all_masks[0]:
        mask_512 = all_masks[0][1].astype(np.uint8)
    else:
        mask_512 = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    mask_pil = Image.fromarray(mask_512 * 255).resize((W, H), Image.NEAREST)
    return (np.array(mask_pil) > 127).astype(np.uint8)


def show_mask(mask, ax):
    color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def show_box(box, ax):
    x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


# =======================================================================
# ========================= PIPELINE ===================================
# =======================================================================

image_paths = sorted(glob.glob(os.path.join(test_folder, "*.png")))
total = len(image_paths)
print(f"\n{'=' * 60}")
print(f"  Processing {total} images  |  YOLO batch={YOLO_BATCH_SIZE}")
print(f"  Output → {output_folder}")
print(f"{'=' * 60}\n")

pipeline_start = time.time()
images_with_detections = 0
total_boxes = 0

# ── Phase 1: Batched YOLO detection ──────────────────────────────
print("[Phase 1] Running batched YOLO detection …")
t1 = time.time()

# We'll store detection results keyed by image path
detection_results = {}  # path → { "left": [...], "right": [...], "W": int, "H": int }

for batch_start in range(0, total, YOLO_BATCH_SIZE):
    batch_paths = image_paths[batch_start : batch_start + YOLO_BATCH_SIZE]
    results = yolo_model.predict(batch_paths, conf=0.1, verbose=False)

    for img_path, result in zip(batch_paths, results):
        with Image.open(img_path) as pil_img:
            img_w, img_h = pil_img.size

        left_boxes = []
        right_boxes = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x_mid = (x1 + x2) / 2.0
            if x_mid < img_w / 2.0:
                left_boxes.append((conf, x1, y1, x2, y2))
            else:
                right_boxes.append((conf, x1, y1, x2, y2))

        left_boxes = filter_boxes_by_confidence(left_boxes, MERGE_DISTANCE_THRESHOLD)
        right_boxes = filter_boxes_by_confidence(right_boxes, MERGE_DISTANCE_THRESHOLD)

        n_boxes = len(left_boxes) + len(right_boxes)
        total_boxes += n_boxes
        if n_boxes > 0:
            images_with_detections += 1

        detection_results[img_path] = {
            "left": left_boxes,
            "right": right_boxes,
            "W": img_w,
            "H": img_h,
        }

    done = min(batch_start + YOLO_BATCH_SIZE, total)
    print(f"  YOLO: {done}/{total} images", end="\r")

print(
    f"\n[Phase 1] Done in {time.time() - t1:.1f}s  |  "
    f"{images_with_detections} images with detections  |  {total_boxes} total boxes"
)


# ── Phase 2: MedSAM2 segmentation (in-process, no subprocess) ───
print(f"\n[Phase 2] Running MedSAM2 segmentation on {images_with_detections} images …")
t2 = time.time()
seg_count = 0

for idx, img_path in enumerate(image_paths):
    det = detection_results[img_path]
    if not det["left"] and not det["right"]:
        continue

    image_name = os.path.splitext(os.path.basename(img_path))[0]
    img_np = io.imread(img_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W = det["H"], det["W"]

    # Prepare the image tensor once per image
    img_tensor = prepare_image_tensor(img_3c)

    def process_knee(knee_boxes, knee_folder):
        """Segment all boxes for one knee side."""
        if not knee_boxes:
            return []

        expanded = [
            expand_box(x1, y1, x2, y2, W, H, ratio=BBOX_EXPANSION_RATIO)
            for _, x1, y1, x2, y2 in knee_boxes
        ]

        out_dir = os.path.join(knee_folder, image_name)
        os.makedirs(out_dir, exist_ok=True)
        seg_files = []

        for i, box in enumerate(expanded):
            box_np = np.array(box, dtype=np.float64)
            mask = medsam2_segment_box(img_tensor, box_np, H, W)

            seg_path = os.path.join(out_dir, f"seg_{image_name}_{i}.png")
            io.imsave(seg_path, mask.astype(np.uint8) * 255, check_contrast=False)
            seg_files.append(seg_path)

            if SAVE_VIS:
                vis_path = os.path.join(out_dir, f"vis_{image_name}_{i}.png")
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(img_3c)
                show_box(box_np, ax[0])
                ax[0].set_title("Input + Box")
                ax[1].imshow(img_3c)
                show_mask(mask, ax[1])
                show_box(box_np, ax[1])
                ax[1].set_title("MedSAM2 Segmentation")
                plt.savefig(vis_path)
                plt.close()

        return seg_files

    left_segs = process_knee(det["left"], left_knee_folder)
    right_segs = process_knee(det["right"], right_knee_folder)
    all_segs = left_segs + right_segs
    seg_count += len(all_segs)

    # ── Merge masks ──
    if all_segs:
        merged_mask = None
        for seg_f in all_segs:
            seg_img = io.imread(seg_f)
            seg_bin = (seg_img > 0).astype(np.uint8)
            merged_mask = (
                seg_bin if merged_mask is None else np.maximum(merged_mask, seg_bin)
            )

        merged_out = os.path.join(merged_folder, f"seg_{image_name}.png")
        io.imsave(merged_out, merged_mask.astype(np.uint8) * 255, check_contrast=False)

    # Progress
    print(
        f"  [{seg_count} masks] {image_name} — "
        f"L:{len(det['left'])} R:{len(det['right'])}  "
        f"({idx + 1}/{total})",
        end="\r",
    )

elapsed = time.time() - pipeline_start
print(f"\n\n[Phase 2] Segmentation done in {time.time() - t2:.1f}s")
print(f"\n{'=' * 60}")
print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
print(f"  Images: {total}  |  With detections: {images_with_detections}")
print(f"  Total masks: {seg_count}")
print(f"  Output: {output_folder}")
print(f"{'=' * 60}")
