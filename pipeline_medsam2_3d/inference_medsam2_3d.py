# -*- coding: utf-8 -*-
"""
inference_medsam2_3d.py — 3D Video-mode MedSAM2 pipeline.

Uses SAM2's video propagation to segment bones across the entire CT volume.
Instead of treating each slice independently (2D mode), this pipeline:

  1. Runs batched YOLO on all slices to get per-slice bounding boxes
  2. Tracks/clusters those boxes across Z into persistent bone tracks
  3. Selects a "key frame" per bone (slice where that bone is most visible)
  4. Feeds the entire slice stack as a "video" into MedSAM2
  5. Prompts only on key frames and lets SAM2 propagate masks across all frames

Because the full 704-frame volume (~2.2 GB of image tensors + features) won't
fit in 8 GB VRAM all at once, we process the volume in overlapping chunks
(e.g. 120 frames with 20-frame overlap).  Within each chunk, each bone track
is prompted on its best key frame and propagated forward + backward.

Output structure is identical to the 2D batched pipeline:
    segmented_…/left_knee/<image_name>/seg_*.png
    segmented_…/right_knee/<image_name>/seg_*.png
    segmented_…/Merged/seg_<image_name>.png
"""

import os
import sys
import glob
import math
import time
from collections import defaultdict

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

PATIENT_ID = os.getenv("PATIENT_ID", "15").strip()
DIRECTION = os.getenv("DIRECTION", "axial").strip().lower()
RUN_SUFFIX = os.getenv("RUN_SUFFIX", "N10_conf05_iou05").strip()

if DIRECTION not in {"axial", "coronal", "sagittal"}:
    sys.exit(
        f"[ERROR] Invalid DIRECTION='{DIRECTION}'. Use one of: axial, coronal, sagittal"
    )

test_folder = os.path.join(
    PROJECT_ROOT, f"pipeline_medsam2_3d/p{PATIENT_ID}_extracted_pngs/{DIRECTION}"
)
output_folder = os.path.join(
    PROJECT_ROOT,
    f"pipeline_medsam2_3d/segmented_ct_slices_p{PATIENT_ID}_3d_{DIRECTION}_{RUN_SUFFIX}/",
)

# ── MedSAM2 paths ──
medsam2_repo = os.path.join(PROJECT_ROOT, "pipeline_medsam2/MedSAM2")
medsam2_checkpoint = "MedSAM2_latest.pt"
medsam2_model_cfg = "configs/sam2.1_hiera_t512.yaml"
medsam2_device = "cuda"

# ── YOLO paths ──
yolo_weights_override = os.getenv("YOLO_WEIGHTS_PATH", "").strip()
if yolo_weights_override:
    yolo_weights_path = yolo_weights_override
else:
    yolo_weights_path = os.path.join(
        PROJECT_ROOT,
        f"pipeline_medsam2_3d/runs_yolo12/exp_{DIRECTION}/weights/last.pt",
    )

# ── Batching / Chunking ──
YOLO_BATCH_SIZE = 128
CHUNK_SIZE = 100  # frames per chunk for video propagation
CHUNK_OVERLAP = 30  # overlapping frames between consecutive chunks

# ── Detection tuning ──
BBOX_EXPANSION_RATIO = 0.06
MERGE_DISTANCE_THRESHOLD = 2
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# ── Bone tracking ──
TRACK_IOU_THRESHOLD = float(
    os.getenv("TRACK_IOU_THRESHOLD", "0.5")
)  # min IoU between boxes on consecutive frames to link
TRACK_GAP_TOLERANCE = 5  # max gap (frames) allowed within a track before splitting

# ── MedSAM2 ──
IMAGE_SIZE = 512  # MedSAM2 internal resolution
SCORE_THRESHOLD = 0.0  # logit threshold for mask output
PROMPT_EVERY_N = 10  # add a box prompt every N YOLO-detected frames per chunk
# (1 = prompt every frame ≈ 2D quality, higher = faster but less accurate)

# ── Visualisation ──
SAVE_VIS = False  # generate per-box overlay PNGs (vis_*)

# =======================================================================
# ========================= SETUP ======================================
# =======================================================================

left_knee_folder = os.path.join(output_folder, "left_knee")
right_knee_folder = os.path.join(output_folder, "right_knee")
merged_folder = os.path.join(output_folder, "Merged")
for d in [left_knee_folder, right_knee_folder, merged_folder]:
    os.makedirs(d, exist_ok=True)

# ── Load MedSAM2 ──
print("[INIT] Loading MedSAM2 video predictor …")
t0 = time.time()

medsam2_root = os.path.abspath(medsam2_repo)
if medsam2_root not in sys.path:
    sys.path.insert(0, medsam2_root)

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

# ── Load YOLO ──
print("[INIT] Loading YOLO model …")
yolo_model = YOLO(yolo_weights_path)
print("[INIT] YOLO loaded ✓")


# =======================================================================
# ========================= HELPERS ====================================
# =======================================================================


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


def box_iou(boxA, boxB):
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def box_area(box):
    """Area of an (x1, y1, x2, y2) box."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def prepare_single_frame(img_3c: np.ndarray) -> torch.Tensor:
    """Resize H×W×3 uint8 → (3, 512, 512) float, ImageNet-normalised."""
    pil = (
        Image.fromarray(img_3c)
        .convert("RGB")
        .resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    )
    arr = np.array(pil).astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return torch.from_numpy((arr - mean) / std)


def show_mask(mask, ax):
    """Overlay a semi-transparent yellow mask on an axes."""
    color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def show_box(box, ax):
    """Draw a blue bounding-box rectangle on an axes."""
    x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


# =======================================================================
# ============= PHASE 1: BATCHED YOLO DETECTION =======================
# =======================================================================

image_paths = sorted(glob.glob(os.path.join(test_folder, "*.png")))
total_frames = len(image_paths)
print(f"\n{'=' * 70}")
print(
    f"  3D Pipeline  |  {total_frames} frames  |  chunk={CHUNK_SIZE}  overlap={CHUNK_OVERLAP}"
)
print(f"  Output → {output_folder}")
print(f"{'=' * 70}\n")

pipeline_start = time.time()

print("[Phase 1] Batched YOLO detection …")
t1 = time.time()

# per_frame_boxes[i] = list of (conf, x1, y1, x2, y2, "left"/"right") for global frame i
per_frame_boxes = [[] for _ in range(total_frames)]
# Also store image dimensions (should all be the same for CT)
img_dims = [None] * total_frames

for batch_start in range(0, total_frames, YOLO_BATCH_SIZE):
    batch_paths = image_paths[batch_start : batch_start + YOLO_BATCH_SIZE]
    results = yolo_model.predict(batch_paths, conf=0.1, verbose=False)

    for j, (img_path, result) in enumerate(zip(batch_paths, results)):
        frame_idx = batch_start + j
        with Image.open(img_path) as pil_img:
            img_w, img_h = pil_img.size
        img_dims[frame_idx] = (img_w, img_h)

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

        for b in left_boxes:
            per_frame_boxes[frame_idx].append((*b, "left"))
        for b in right_boxes:
            per_frame_boxes[frame_idx].append((*b, "right"))

    done = min(batch_start + YOLO_BATCH_SIZE, total_frames)
    print(f"  YOLO: {done}/{total_frames}", end="\r")

n_det = sum(len(b) for b in per_frame_boxes)
n_frames_det = sum(1 for b in per_frame_boxes if b)
print(
    f"\n[Phase 1] Done in {time.time() - t1:.1f}s  |  {n_frames_det} frames with detections  |  {n_det} total boxes"
)


# =======================================================================
# ============= PHASE 2: BONE TRACKING / CLUSTERING ===================
# =======================================================================

print("\n[Phase 2] Tracking bones across slices …")
t2 = time.time()


class BoneTrack:
    """Represents a single bone tracked across multiple Z-slices."""

    def __init__(self, track_id, side, first_frame, first_box):
        self.track_id = track_id
        self.side = side  # "left" or "right"
        self.frames = {first_frame: first_box}  # frame_idx → (x1, y1, x2, y2)
        self.last_frame = first_frame

    @property
    def start_frame(self):
        return min(self.frames.keys())

    @property
    def end_frame(self):
        return max(self.frames.keys())

    @property
    def key_frame(self):
        """Frame where the bone has the largest bounding box (most visible)."""
        return max(self.frames.keys(), key=lambda f: box_area(self.frames[f]))

    @property
    def key_box(self):
        return self.frames[self.key_frame]

    def try_extend(self, frame_idx, box_xyxy, iou_thresh, gap_tolerance):
        """Try to extend this track with a new detection. Returns True if linked."""
        if frame_idx - self.last_frame > gap_tolerance:
            return False
        last_box = self.frames[self.last_frame]
        iou = box_iou(last_box, box_xyxy)
        if iou >= iou_thresh:
            self.frames[frame_idx] = box_xyxy
            self.last_frame = frame_idx
            return True
        return False


# Build tracks by greedy assignment: for each frame, try to extend existing
# active tracks; if no match, start a new track.
next_track_id = 0
active_tracks = []  # tracks that might still be extended
finished_tracks = []  # tracks that have been closed

for frame_idx in range(total_frames):
    detections = per_frame_boxes[frame_idx]
    if not detections:
        continue

    # Expand boxes for matching but track original expanded boxes
    img_w, img_h = img_dims[frame_idx]

    # Group detections by side
    for det in detections:
        conf, x1, y1, x2, y2, side = det
        ex_box = expand_box(x1, y1, x2, y2, img_w, img_h, BBOX_EXPANSION_RATIO)
        box_xyxy = tuple(ex_box)

        # Try to extend an existing active track (same side only)
        matched = False
        for track in active_tracks:
            if track.side != side:
                continue
            if track.try_extend(
                frame_idx, box_xyxy, TRACK_IOU_THRESHOLD, TRACK_GAP_TOLERANCE
            ):
                matched = True
                break

        if not matched:
            # Start a new track
            new_track = BoneTrack(next_track_id, side, frame_idx, box_xyxy)
            active_tracks.append(new_track)
            next_track_id += 1

    # Retire tracks that have been inactive too long
    still_active = []
    for track in active_tracks:
        if frame_idx - track.last_frame > TRACK_GAP_TOLERANCE:
            finished_tracks.append(track)
        else:
            still_active.append(track)
    active_tracks = still_active

# Finalize remaining active tracks
finished_tracks.extend(active_tracks)
active_tracks = []

# Filter out very short tracks (noise) — must span at least 3 frames
all_tracks = [t for t in finished_tracks if len(t.frames) >= 3]
all_tracks.sort(key=lambda t: (t.side, t.start_frame))

print(f"  Found {len(all_tracks)} bone tracks:")
for t in all_tracks:
    span = t.end_frame - t.start_frame + 1
    print(
        f"    Track {t.track_id:3d} [{t.side:5s}]  "
        f"frames {t.start_frame:4d}–{t.end_frame:4d}  "
        f"({len(t.frames):3d} detections in {span} span)  "
        f"key_frame={t.key_frame}"
    )
print(f"[Phase 2] Done in {time.time() - t2:.1f}s")


# =======================================================================
# ============= PHASE 3: CHUNKED VIDEO PROPAGATION ====================
# =======================================================================

print(
    f"\n[Phase 3] MedSAM2 3D video propagation (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) …"
)
t3 = time.time()

# Pre-load all images and prepare tensors (we'll load per-chunk to save RAM)
# Actually, let's prepare a function that loads a chunk of images as tensors.


def load_chunk_tensors(start, end):
    """Load frames [start, end) as a stacked (N, 3, 512, 512) tensor on CPU."""
    tensors = []
    for i in range(start, end):
        img = io.imread(image_paths[i])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        tensors.append(prepare_single_frame(img))
    return torch.stack(tensors, dim=0)  # (N, 3, 512, 512)


# Final output: per-frame, per-obj_id mask storage
# We'll build a dict: global_frame_idx → {obj_id → binary_mask_HxW}
frame_masks = defaultdict(dict)  # frame_idx → {track_id → mask(H, W) bool}

# Determine chunks
chunk_starts = list(range(0, total_frames, CHUNK_SIZE - CHUNK_OVERLAP))
# Make sure the last chunk covers the end
if chunk_starts[-1] + CHUNK_SIZE < total_frames:
    chunk_starts.append(max(0, total_frames - CHUNK_SIZE))
# Remove duplicates and sort
chunk_starts = sorted(set(chunk_starts))

print(f"  Processing {len(chunk_starts)} chunks …")

for chunk_idx, chunk_start in enumerate(chunk_starts):
    chunk_end = min(chunk_start + CHUNK_SIZE, total_frames)
    chunk_len = chunk_end - chunk_start
    img_w, img_h = img_dims[chunk_start] if img_dims[chunk_start] else (512, 512)

    # Find which tracks have key frames (or any frames) in this chunk
    chunk_tracks = []
    for track in all_tracks:
        # A track is relevant to this chunk if its key frame falls within it
        # OR if it overlaps with this chunk's range and we should prompt it
        if track.start_frame < chunk_end and track.end_frame >= chunk_start:
            # Does the key frame fall in this chunk?
            if chunk_start <= track.key_frame < chunk_end:
                chunk_tracks.append(track)
            # If key frame is outside but track spans this chunk, we need to
            # find the best frame within this chunk to use as prompt
            elif any(chunk_start <= f < chunk_end for f in track.frames):
                # Use the frame with the largest box within this chunk
                chunk_frames = {
                    f: b
                    for f, b in track.frames.items()
                    if chunk_start <= f < chunk_end
                }
                if chunk_frames:
                    chunk_tracks.append(track)

    if not chunk_tracks:
        print(
            f"  Chunk {chunk_idx + 1}/{len(chunk_starts)} [{chunk_start}–{chunk_end}): no tracks → skip"
        )
        continue

    # Load chunk images
    print(
        f"  Chunk {chunk_idx + 1}/{len(chunk_starts)} [{chunk_start}–{chunk_end}): "
        f"{len(chunk_tracks)} tracks, loading images …",
        end="",
    )
    chunk_tensor = load_chunk_tensors(chunk_start, chunk_end)
    chunk_tensor_gpu = chunk_tensor.to(medsam2_device)
    print(f" {chunk_tensor.shape[0]} frames loaded")

    # Process each bone track separately to handle appearing/disappearing bones
    # (similar to vos_separate_inference_per_object)
    chunk_scores_per_track = {}  # track_id → {local_frame_idx → mask_logits}

    for track in chunk_tracks:
        # ── Collect ALL YOLO-detected frames for this track within this chunk ──
        chunk_frames = {
            f: b for f, b in track.frames.items() if chunk_start <= f < chunk_end
        }
        if not chunk_frames:
            continue

        # Select prompt frames: every PROMPT_EVERY_N-th YOLO frame in the chunk
        sorted_det_frames = sorted(chunk_frames.keys())
        prompt_global_frames = sorted_det_frames[::PROMPT_EVERY_N]
        # Always include the first and last detection frames for best coverage
        if sorted_det_frames[0] not in prompt_global_frames:
            prompt_global_frames.insert(0, sorted_det_frames[0])
        if sorted_det_frames[-1] not in prompt_global_frames:
            prompt_global_frames.append(sorted_det_frames[-1])

        with (
            torch.inference_mode(),
            torch.autocast(medsam2_device, dtype=torch.bfloat16),
        ):
            inference_state = predictor.init_state(
                chunk_tensor_gpu,
                img_h,
                img_w,
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
            )

            # Add box prompts on MULTIPLE frames, with negative points
            # from other tracks' boxes to prevent leaking into neighboring bones
            for gf in prompt_global_frames:
                local_f = gf - chunk_start
                box_for_prompt = np.array(chunk_frames[gf], dtype=np.float64)

                # Collect centres of OTHER tracks' boxes on this frame as negatives
                neg_pts = []
                for other_track in chunk_tracks:
                    if other_track.track_id == track.track_id:
                        continue
                    if gf in other_track.frames:
                        ob = other_track.frames[gf]
                        cx = (ob[0] + ob[2]) / 2
                        cy = (ob[1] + ob[3]) / 2
                        neg_pts.append([cx, cy])

                pts_arg = None
                lbls_arg = None
                if neg_pts:
                    pts_arg = np.array(neg_pts, dtype=np.float32)
                    lbls_arg = np.zeros(len(neg_pts), dtype=np.int32)  # 0 = negative

                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=local_f,
                    obj_id=track.track_id,
                    box=box_for_prompt,
                    points=pts_arg,
                    labels=lbls_arg,
                )

            # Propagate forward through the entire chunk (from frame 0)
            # SAM2 uses all registered prompts as it reaches each frame
            track_masks = {}
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                reverse=False,
            ):
                mask = (out_mask_logits[0] > SCORE_THRESHOLD).cpu().numpy().squeeze()
                track_masks[out_frame_idx] = mask

            predictor.reset_state(inference_state)

        chunk_scores_per_track[track.track_id] = track_masks

    # Merge chunk results into global frame_masks
    # For overlapping regions, we take the union of masks
    for track_id, local_masks in chunk_scores_per_track.items():
        track = next(t for t in all_tracks if t.track_id == track_id)
        for local_idx, mask in local_masks.items():
            global_idx = chunk_start + local_idx
            # Only store masks where the bone was actually detected by YOLO
            # (or within a small margin) to avoid hallucinated propagation
            has_yolo = global_idx in track.frames
            near_yolo = any(
                abs(global_idx - f) <= TRACK_GAP_TOLERANCE for f in track.frames
            )
            if has_yolo or near_yolo:
                if track_id in frame_masks[global_idx]:
                    # Union with existing (from overlap region)
                    frame_masks[global_idx][track_id] = (
                        frame_masks[global_idx][track_id] | mask
                    )
                else:
                    frame_masks[global_idx][track_id] = mask

    # Free GPU memory
    del chunk_tensor_gpu, chunk_tensor
    torch.cuda.empty_cache()

n_frames_with_masks = sum(1 for fm in frame_masks.values() if fm)
n_total_masks = sum(len(fm) for fm in frame_masks.values())
print(
    f"\n[Phase 3] Done in {time.time() - t3:.1f}s  |  "
    f"{n_frames_with_masks} frames with masks  |  {n_total_masks} total track-masks"
)


# =======================================================================
# ============= PHASE 4: SAVE OUTPUT MASKS =============================
# =======================================================================

print(f"\n[Phase 4] Saving output masks …")
t4 = time.time()

seg_count = 0
merged_count = 0

for frame_idx in range(total_frames):
    track_masks = frame_masks.get(frame_idx, {})
    if not track_masks:
        continue

    image_name = os.path.splitext(os.path.basename(image_paths[frame_idx]))[0]
    img_w, img_h = img_dims[frame_idx]

    # Group masks by side
    left_masks = []
    right_masks = []
    for track_id, mask in track_masks.items():
        track = next(t for t in all_tracks if t.track_id == track_id)
        # Resize mask from model output size to original image size
        if mask.shape != (img_h, img_w):
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
            mask = (np.array(mask_pil) > 127).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        if track.side == "left":
            left_masks.append((track_id, mask))
        else:
            right_masks.append((track_id, mask))

    # Load original image once (only if we need vis)
    if SAVE_VIS:
        img_3c = np.array(Image.open(image_paths[frame_idx]).convert("RGB"))

    # Save per-side masks
    for side, masks, knee_folder in [
        ("left", left_masks, left_knee_folder),
        ("right", right_masks, right_knee_folder),
    ]:
        if not masks:
            continue
        out_dir = os.path.join(knee_folder, image_name)
        os.makedirs(out_dir, exist_ok=True)
        for i, (tid, mask) in enumerate(masks):
            seg_path = os.path.join(out_dir, f"seg_{image_name}_{i}.png")
            io.imsave(seg_path, mask * 255, check_contrast=False)
            seg_count += 1

            if SAVE_VIS:
                track = next(t for t in all_tracks if t.track_id == tid)
                # Get the bounding box for this frame (or nearest)
                box_np = None
                if frame_idx in track.frames:
                    box_np = np.array(track.frames[frame_idx], dtype=np.float64)
                else:
                    # Find nearest frame with a box
                    nearest = min(track.frames.keys(), key=lambda f: abs(f - frame_idx))
                    box_np = np.array(track.frames[nearest], dtype=np.float64)

                vis_path = os.path.join(out_dir, f"vis_{image_name}_{i}.png")
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(img_3c)
                show_box(box_np, ax[0])
                ax[0].set_title(f"Input + Box (track {tid})")
                ax[0].axis("off")
                ax[1].imshow(img_3c)
                show_mask(mask, ax[1])
                show_box(box_np, ax[1])
                ax[1].set_title("MedSAM2 Segmentation")
                ax[1].axis("off")
                plt.tight_layout()
                plt.savefig(vis_path, dpi=100)
                plt.close(fig)

    # Merge all masks
    all_frame_masks = left_masks + right_masks
    if all_frame_masks:
        merged = np.zeros((img_h, img_w), dtype=np.uint8)
        for _, mask in all_frame_masks:
            merged = np.maximum(merged, mask)
        merged_path = os.path.join(merged_folder, f"seg_{image_name}.png")
        io.imsave(merged_path, merged * 255, check_contrast=False)
        merged_count += 1

    if merged_count % 50 == 0:
        print(f"  Saved {merged_count} merged masks …", end="\r")

print(
    f"\n[Phase 4] Saved {seg_count} per-bone masks, {merged_count} merged masks in {time.time() - t4:.1f}s"
)


# =======================================================================
# ========================= SUMMARY ====================================
# =======================================================================

elapsed = time.time() - pipeline_start
print(f"\n{'=' * 70}")
print(f"  ✅ 3D Pipeline complete in {elapsed:.1f}s")
print(f"  Frames: {total_frames}  |  With detections: {n_frames_det}")
print(f"  Bone tracks: {len(all_tracks)}")
print(f"  Chunks processed: {len(chunk_starts)}")
print(f"  Merged masks: {merged_count}")
print(f"  Per-bone masks: {seg_count}")
print(f"  Output: {output_folder}")
print(f"{'=' * 70}")
