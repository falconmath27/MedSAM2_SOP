# Pipeline: YOLO + MedSAM2 (Knee CT Bone Segmentation)

This pipeline replaces the original MedSAM with **MedSAM2** (`wanglab/MedSAM2`)
for the segmentation step. MedSAM2 is built on SAM 2.1 and is designed for 3D
medical image and video segmentation, but we use it here in a 2D slice-by-slice
mode (treating slices as single-frame "videos").

## Setup

### 1. Install MedSAM2

```bash
conda create -n medsam2 python=3.12 -y
conda activate medsam2

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Clone and install MedSAM2
git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2
pip install -e ".[dev]"
cd ..

# Additional dependencies
pip install ultralytics scikit-image Pillow matplotlib
```

### 2. Download the MedSAM2 checkpoint

```bash
cd MedSAM2
bash download.sh   # downloads all checkpoints to MedSAM2/checkpoints/
cd ..
```

Or download just the recommended model:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="wanglab/MedSAM2", filename="MedSAM2_latest.pt",
                local_dir="MedSAM2/checkpoints")
```

### 3. Configure paths

Edit the `# ===== USER CONFIG =====` section at the top of each script to point
to your local paths.

## Usage

### Step 1: Run the full YOLO → MedSAM2 pipeline

```bash
python inference_medsam2.py
```

This will:

1. Run YOLO on each PNG slice to detect bone bounding boxes.
2. Call `MedSAM2_Inference.py` as a subprocess for each knee.
3. Merge left/right masks into a combined segmentation per slice.

### Step 2 (standalone): Run MedSAM2 segmentation on a single image

```bash
python MedSAM2_Inference.py \
    -i path/to/slice.png \
    -o path/to/output/ \
    --box "[[x1,y1,x2,y2], [x1,y1,x2,y2]]" \
    --medsam2_repo /path/to/MedSAM2
```

## Key Differences from Original MedSAM Pipeline

| Aspect | Original (MedSAM) | This Pipeline (MedSAM2) |
|--------|-------------------|------------------------|
| Model | MedSAM (ViT-B SAM fine-tuned) | MedSAM2 (SAM 2.1 Hiera fine-tuned) |
| Input size | 1024×1024 | 512×512 |
| Architecture | SAM encoder/decoder | SAM2 video predictor (used per-frame) |
| Checkpoint | `medsam_vit_b.pth` | `MedSAM2_latest.pt` |
| CT-specific | Generic medical | Has CT-specific model (`MedSAM2_CTLesion.pt`) |
