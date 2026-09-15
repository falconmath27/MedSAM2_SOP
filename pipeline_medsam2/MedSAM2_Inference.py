# -*- coding: utf-8 -*-
"""
MedSAM2_Inference.py — Drop-in replacement for MedSAM_Inference.py using MedSAM2.

MedSAM2 is built on SAM 2.1 and uses a *video predictor* architecture.
For 2D images we treat the single slice as a 1-frame "video" and prompt
with a bounding box on frame 0, then read back the mask.

Usage (same CLI interface as the original):
    python MedSAM2_Inference.py \
        -i <image.png> \
        -o <output_dir> \
        --box "[[x1,y1,x2,y2], ...]" \
        --medsam2_repo /path/to/MedSAM2   \
        [--checkpoint MedSAM2_latest.pt]   \
        [--device cpu|cuda]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import torch

# ─── CLI ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run MedSAM2 inference on a 2D image with box prompts"
)
parser.add_argument(
    "-i", "--data_path", type=str, required=True, help="Path to input PNG image"
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    required=True,
    help="Output folder for segmentation masks",
)
parser.add_argument(
    "--box",
    type=str,
    required=True,
    help='Bounding boxes as string, e.g. "[[x1,y1,x2,y2]]"',
)
parser.add_argument("--device", type=str, default="cpu", help="Device: cpu | cuda")
parser.add_argument(
    "--medsam2_repo",
    type=str,
    default=os.environ.get("MEDSAM2_REPO", "MedSAM2"),
    help="Path to the cloned bowang-lab/MedSAM2 repo",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="MedSAM2_latest.pt",
    help="Checkpoint filename inside <medsam2_repo>/checkpoints/",
)
parser.add_argument(
    "--model_cfg",
    type=str,
    default="configs/sam2.1_hiera_t512.yaml",
    help="Model config (relative to MedSAM2 repo)",
)
args = parser.parse_args()

# ─── Add MedSAM2 to sys.path so we can import sam2.* ───────────────
medsam2_root = os.path.abspath(args.medsam2_repo)
if medsam2_root not in sys.path:
    sys.path.insert(0, medsam2_root)

# Hydra needs to resolve configs relative to MedSAM2 repo
original_cwd = os.getcwd()
os.chdir(medsam2_root)

from sam2.build_sam import build_sam2_video_predictor_npz  # noqa: E402

# ─── Resolve checkpoint path ───────────────────────────────────────
ckpt_path = os.path.join(medsam2_root, "checkpoints", args.checkpoint)
if not os.path.isfile(ckpt_path):
    # Try as absolute / relative path directly
    ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path):
        sys.exit(f"[ERROR] Checkpoint not found: {ckpt_path}")

model_cfg = args.model_cfg
device = args.device

print(f"[MedSAM2] Loading model  cfg={model_cfg}  ckpt={ckpt_path}  device={device}")

# ─── Helpers ───────────────────────────────────────────────────────

IMAGE_SIZE = 512  # MedSAM2 uses 512×512 internally


def show_mask(mask, ax, random_color=False):
    color = (
        np.concatenate([np.random.random(3), [0.6]])
        if random_color
        else np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    )
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


def prepare_image_tensor(img_3c: np.ndarray) -> torch.Tensor:
    """
    Resize an H×W×3 uint8 image to (1, 3, 512, 512) float tensor,
    normalised with ImageNet mean/std (as MedSAM2 expects).
    """
    pil = (
        Image.fromarray(img_3c)
        .convert("RGB")
        .resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    )
    arr = np.array(pil).astype(np.float32).transpose(2, 0, 1) / 255.0  # (3,512,512)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).unsqueeze(0)  # (1, 3, 512, 512)


def medsam2_inference_2d(predictor, img_tensor, box_np, H, W, device):
    """
    Run MedSAM2 on a single 2D image (treated as a 1-frame video).

    Parameters
    ----------
    predictor : SAM2VideoPredictorNPZ
    img_tensor : Tensor (1, 3, 512, 512) normalised
    box_np : ndarray (4,) — [x1, y1, x2, y2] in *original* pixel coords
    H, W : original image dimensions
    device : str

    Returns
    -------
    mask : ndarray (H, W) uint8 binary {0, 1}
    """
    img_tensor = img_tensor.to(device)

    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.bfloat16, enabled=(device != "cpu")),
    ):
        inference_state = predictor.init_state(img_tensor, H, W)

        # Scale box from original coords → 512×512 coords for the predictor
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

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=box_scaled,
        )

        # Propagate (single frame — just finalises the mask)
        all_masks = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            all_masks[frame_idx] = {
                oid: (mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, oid in enumerate(obj_ids)
            }

        # Reset state for next call
        predictor.reset_state(inference_state)

    # Retrieve mask for frame 0, object 1
    if 0 in all_masks and 1 in all_masks[0]:
        mask_512 = all_masks[0][1].astype(np.uint8)
    else:
        mask_512 = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    # Resize mask back to original resolution
    mask_pil = Image.fromarray(mask_512 * 255).resize((W, H), Image.NEAREST)
    mask_orig = (np.array(mask_pil) > 127).astype(np.uint8)
    return mask_orig


# ─── Main ──────────────────────────────────────────────────────────


def main():
    # Build predictor
    predictor = build_sam2_video_predictor_npz(model_cfg, ckpt_path, device=device)

    # Load image
    img_np = io.imread(args.data_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape

    # Prepare image tensor once
    img_tensor = prepare_image_tensor(img_3c)

    # Parse bounding boxes
    box_list = eval(args.box)  # e.g. [[x1,y1,x2,y2], ...]
    if not isinstance(box_list[0], list):
        box_list = [box_list]

    os.makedirs(args.seg_path, exist_ok=True)
    image_name = os.path.basename(args.data_path).replace(".png", "")
    seg_output_files = []

    for i, box in enumerate(box_list):
        box_np = np.array(box, dtype=np.float64)

        mask = medsam2_inference_2d(predictor, img_tensor, box_np, H, W, device)

        # Save per-box segmentation
        seg_out_path = os.path.join(args.seg_path, f"seg_{image_name}_{i}.png")
        seg_output_files.append(seg_out_path)
        io.imsave(seg_out_path, mask.astype(np.uint8) * 255, check_contrast=False)
        print(f"  Saved segmentation: {seg_out_path}")

        # Save visualisation
        vis_out_path = os.path.join(args.seg_path, f"vis_{image_name}_{i}.png")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        show_box(box_np, ax[0])
        ax[0].set_title("Input Image + Box")
        ax[1].imshow(img_3c)
        show_mask(mask, ax[1])
        show_box(box_np, ax[1])
        ax[1].set_title("MedSAM2 Segmentation")
        plt.savefig(vis_out_path)
        plt.close()
        print(f"  Saved visualisation: {vis_out_path}")

    # ── Merge all per-box masks into one ──
    if seg_output_files:
        merged_mask = np.zeros((H, W), dtype=np.uint8)
        for seg_path in seg_output_files:
            seg_img = io.imread(seg_path)
            merged_mask = np.maximum(merged_mask, (seg_img > 0).astype(np.uint8))

        merged_out = os.path.join(args.seg_path, f"seg_{image_name}.png")
        io.imsave(merged_out, merged_mask * 255, check_contrast=False)
        print(f"  Saved merged segmentation: {merged_out}")


if __name__ == "__main__":
    main()
