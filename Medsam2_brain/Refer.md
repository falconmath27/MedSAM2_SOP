```markdown
# System Architecture & Implementation Plan: `Auto-MedSAM2-CT`

> **Project Title:** *Auto-MedSAM2-CT: Atlas-Distilled Spatial Prompting of 3D Medical Foundation Models for Automated Brain Territory Parcellation and Stroke Localization on Non-Contrast CT*  
> **Dataset:** APIS (A Paired CT-MRI Dataset for Ischemic Stroke Segmentation, $N = 60$ paired cases)  
> **Core Model:** MedSAM2 (`MedSAM2_latest.pt`, `sam2.1_hiera_t512`) — Inference-Only Pipeline  
> **Target Output:** 3D Multi-Region NIfTI segmentation maps (`.nii.gz`) with integer labels `0` (Background) through `12` (Bilateral Stroke Territories).

---

## 1. End-to-End System Architecture Diagram

```text
========================================================================================================
               PHASE 1: OFFLINE CROSS-MODAL REFERENCE & PROMPT DISTILLATION (TRAINING / SETUP)
========================================================================================================

  [3D MNI152 Template] + [Harvard-Oxford Cortical & Subcortical 3D Atlases]
                                      │
                                      ▼ (Merge & Split by Hemisphere X-Midline)
               [Master 12-Region 3D Reference Atlas (Labels 0..12)]
                                      │
                                      ▼ (3D SyN Deformable Registration via ANTsPy)
  [APIS Paired ADC-MRI (N=60)] ──────► [Patient-Specific 3D MRI Silver-Standard Masks (G)]
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
     (Extract 3D Prompts: z_min, z_max,        (Serves as 3D Ground Truth G
      z_key, 2D Box [x1,y1,x2,y2])              for Final CT Evaluation)
                   │
                   ▼
     [Train 5-Fold YOLOv11s Detector on 3-Channel Multi-Window CT Slices]

========================================================================================================
               PHASE 2: TEST-TIME CT-ONLY INFERENCE PIPELINE (ZERO HUMAN INTERVENTION)
========================================================================================================

  [Raw Patient NCCT (.nii.gz)]
        │
        ├──► MODULE 1: 3-Channel Multi-Window Hounsfield Encoding
        │      ├─► Channel R (Stroke Window): WL=35, WW=40  [15..55 HU]  -> Cortical (M1, Insula)
        │      ├─► Channel G (Brain Window) : WL=40, WW=80  [ 0..80 HU]  -> Deep Gray (C, L, Thalamus)
        │      └─► Channel B (CSF Window)   : WL=20, WW=40  [ 0..40 HU]  -> Ventricles & Sulci
        │
        ├──► MODULE 2: Automated 3D Spatial Prompt Generator (Dual-Mode Benchmark)
        │      ├─► Mode A (Atlas-MedSAM2): 3D NCCT Template Registration -> {z_min, z_max, z_key, Box_k}
        │      └─► Mode B (YOLO-MedSAM2) : YOLOv11s Slice Detection      -> {z_min, z_max, z_key, Box_k}
        │
        ├──► MODULE 3: Z-Clamped Bidirectional MedSAM2 Memory Propagation
        │      ├─► Crop 3D Tensor strictly to V[z_min : z_max + 1] (Eliminates Slice Bleeding)
        │      ├─► Prompt MedSAM2 on local slice (z_key - z_min) with 2D Box_k
        │      ├─► Propagate Forward (z_key -> z_max) & Backward (z_key -> z_min)
        │      └─► Store raw 3D spatial probability logits L_k(z, y, x) for k = 1..12
        │
        ├──► MODULE 4: Foreground-Thresholded Logit Argmax (Boundary Conflict Resolution)
        │      └─► Y_hat(z,y,x) = argmax_k L_k(z,y,x) if max_k(L_k) > 0 else 0
        │
        └──► MODULE 5: Evaluation & Automated Clinical Stroke Territory Reporting
               ├─► Compare Y_hat (CT) vs. G (MRI Silver-Standard) -> 3D DSC, IoU, HD95 (mm)
               └─► Intersect Y_hat with APIS Expert Stroke Mask   -> Infarct Burden % per Region
========================================================================================================
```

---

## 2. Project Directory & Repository Structure

Organize your workspace with this modular structure so every experiment, ablation, and figure is reproducible:

```text
Auto-MedSAM2-CT/
├── data/
│   ├── APIS_raw/                        # Original 60 APIS patients (NCCT, ADC, lesion masks)
│   │   ├── train_001/
│   │   │   ├── train_001_ncct.nii.gz
│   │   │   ├── train_001_adc.nii.gz
│   │   │   └── train_001_mask.nii.gz
│   ├── master_atlas_3d/                 # Generated once in Step 1
│   │   ├── mni_template_1mm.nii.gz
│   │   └── stroke_12regions_atlas.nii.gz
│   ├── silver_standard_mri_gt/          # 3D Reference masks (0..12) warped to each patient
│   ├── prompts_json/                    # Saved auto-prompts {patient_id: {region_id: prompt}}
│   │   ├── modeA_atlas_prompts.json
│   │   └── modeB_yolo_prompts.json
│   └── yolo_5fold_dataset/              # 2D Multi-Window PNGs + YOLO .txt labels (Folds 0..4)
├── checkpoints/
│   ├── MedSAM2_latest.pt                # Official Bo Wang Lab MedSAM2 weights
│   └── yolov11s_fold0..4_best.pt        # Trained YOLOv11 weights per fold
├── src/
│   ├── step1_build_12region_atlas.py    # Builds bilateral 12-region MNI atlas via nilearn
│   ├── step2_generate_mri_gt_and_boxes.py # Warps atlas to ADC-MRI & extracts reference boxes
│   ├── step3_multiwindow_ct_encoder.py  # 3-Channel (Stroke/Brain/CSF) CT preprocessor
│   ├── step4_prompt_generators.py       # Mode A (ANTsPy CT warp) & Mode B (YOLOv11s)
│   ├── step5_medsam2_3d_inference.py    # Z-Clamped propagation + Logit Argmax merger
│   └── step6_evaluate_and_ablate.py     # Computes DSC, IoU, HD95, Wilcoxon p-values & Stroke F1
├── outputs/
│   ├── predictions_modeA_atlas_medsam2/ # Final .nii.gz 3D labeled brains (Mode A)
│   ├── predictions_modeB_yolo_medsam2/  # Final .nii.gz 3D labeled brains (Mode B)
│   ├── ablation_outputs/                # A1..A4 variant .nii.gz outputs
│   └── tables_and_figures/              # CSV results, boxplots, 4-panel visual overlays
└── requirements.txt
```

---

## 3. Target Anatomical Regions Specification (`0` to `12`)

The Harvard-Oxford Cortical (`cort-maxprob-thr25-1mm`) and Subcortical (`sub-maxprob-thr25-1mm`) atlases in `nilearn` provide exact anatomical definitions. Because the cortical atlas is bilateral by default, **we split cortical labels across the sagittal midline ($X = \text{center}$)** to obtain distinct Left and Right hemisphere labels:

| Final Label ID | Region Name | Source Atlas & Index | Clinical Role in Acute Stroke CT | Primary HU Window Channel |
| :---: | :--- | :--- | :--- | :--- |
| **`0`** | **Background / Skull / CSF** | Outside Labels `1..12` | Non-target background voxels | — |
| **`1`** | **Precentral Gyrus Left (`M1_L`)** | HO-Cortical ID `7` ($X < X_{\text{mid}}$) | Primary Motor Cortex (Left MCA Territory) | Ch R (`WL=35, WW=40`) |
| **`2`** | **Precentral Gyrus Right (`M1_R`)** | HO-Cortical ID `7` ($X > X_{\text{mid}}$) | Primary Motor Cortex (Right MCA Territory) | Ch R (`WL=35, WW=40`) |
| **`3`** | **Insular Cortex Left (`I_L`)** | HO-Cortical ID `2` ($X < X_{\text{mid}}$) | Insular Ribbon (Early MCA Infarct Sign) | Ch R (`WL=35, WW=40`) |
| **`4`** | **Insular Cortex Right (`I_R`)** | HO-Cortical ID `2` ($X > X_{\text{mid}}$) | Insular Ribbon (Early MCA Infarct Sign) | Ch R (`WL=35, WW=40`) |
| **`5`** | **Caudate Nucleus Left (`C_L`)** | HO-Subcortical ID `5` | Basal Ganglia Head (Lenticulostriate Territory) | Ch G (`WL=40, WW=80`) |
| **`6`** | **Caudate Nucleus Right (`C_R`)** | HO-Subcortical ID `16` | Basal Ganglia Head (Lenticulostriate Territory) | Ch G (`WL=40, WW=80`) |
| **`7`** | **Lentiform Nucleus Left (`L_L`)** | HO-Subcortical ID `6` (Putamen) + `7` (Pallidum) | Deep MCA Ganglionic Territory (Left) | Ch G (`WL=40, WW=80`) |
| **`8`** | **Lentiform Nucleus Right (`L_R`)** | HO-Subcortical ID `17` (Putamen) + `18` (Pallidum) | Deep MCA Ganglionic Territory (Right) | Ch G (`WL=40, WW=80`) |
| **`9`** | **Thalamus Left (`Thal_L`)** | HO-Subcortical ID `4` | Deep Sensory Relay / PCA Territory (Left) | Ch G (`WL=40, WW=80`) |
| **`10`** | **Thalamus Right (`Thal_R`)** | HO-Subcortical ID `15` | Deep Sensory Relay / PCA Territory (Right) | Ch G (`WL=40, WW=80`) |
| **`11`** | **Lateral Ventricle Left (`Vent_L`)** | HO-Subcortical ID `3` | CSF Landmark / Midline Shift Reference (Left) | Ch B (`WL=20, WW=40`) |
| **`12`** | **Lateral Ventricle Right (`Vent_R`)** | HO-Subcortical ID `14` | CSF Landmark / Midline Shift Reference (Right) | Ch B (`WL=20, WW=40`) |

---

## 4. Phase-by-Phase Implementation Code

### Phase 1 (`src/step1_build_12region_atlas.py`): Build the Master 12-Region 3D Atlas
Run this once to automatically download the MNI152 template and merge the Harvard-Oxford Cortical and Subcortical atlases into your single `stroke_12regions_atlas.nii.gz` file with integer IDs `0` through `12`.

```python
import os
import numpy as np
import nibabel as nib
from nilearn import datasets, image

def build_master_12region_atlas(out_dir="data/master_atlas_3d"):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load 1mm MNI152 Template
    mni_tmpl = datasets.load_mni152_template(resolution=1)
    nib.save(mni_tmpl, os.path.join(out_dir, "mni_template_1mm.nii.gz"))
    
    # 2. Fetch Harvard-Oxford Cortical and Subcortical 1mm MaxProb Atlases
    cort_ds = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
    sub_ds  = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')
    
    cort_img = image.resample_to_img(cort_ds.maps, mni_tmpl, interpolation='nearest')
    sub_img  = image.resample_to_img(sub_ds.maps, mni_tmpl, interpolation='nearest')
    
    cort_vol = np.asarray(cort_img.dataobj, dtype=np.int32)
    sub_vol  = np.asarray(sub_img.dataobj, dtype=np.int32)
    
    master_atlas = np.zeros_like(cort_vol, dtype=np.uint8)
    x_mid = cort_vol.shape[0] // 2  # Sagittal midline in MNI space
    
    # 3. Assign Cortical Regions (Split Left vs. Right across X midline)
    # Precentral Gyrus (M1) = Cortical ID 7
    master_atlas[(cort_vol == 7) & (np.arange(cort_vol.shape[0])[:, None, None] < x_mid)] = 1  # M1_L
    master_atlas[(cort_vol == 7) & (np.arange(cort_vol.shape[0])[:, None, None] >= x_mid)] = 2 # M1_R
    
    # Insular Cortex (I) = Cortical ID 2
    master_atlas[(cort_vol == 2) & (np.arange(cort_vol.shape[0])[:, None, None] < x_mid)] = 3  # I_L
    master_atlas[(cort_vol == 2) & (np.arange(cort_vol.shape[0])[:, None, None] >= x_mid)] = 4 # I_R
    
    # 4. Assign Subcortical Regions (Already Lateralized in Harvard-Oxford Subcortical)
    master_atlas[sub_vol == 5]  = 5  # Caudate Left (C_L)
    master_atlas[sub_vol == 16] = 6  # Caudate Right (C_R)
    
    master_atlas[np.isin(sub_vol, [6, 7])]   = 7  # Lentiform Left (Putamen 6 + Pallidum 7)
    master_atlas[np.isin(sub_vol, [17, 18])] = 8  # Lentiform Right (Putamen 17 + Pallidum 18)
    
    master_atlas[sub_vol == 4]  = 9  # Thalamus Left
    master_atlas[sub_vol == 15] = 10 # Thalamus Right
    
    master_atlas[sub_vol == 3]  = 11 # Lateral Ventricle Left
    master_atlas[sub_vol == 14] = 12 # Lateral Ventricle Right
    
    out_nii = nib.Nifti1Image(master_atlas, affine=mni_tmpl.affine, header=mni_tmpl.header)
    out_path = os.path.join(out_dir, "stroke_12regions_atlas.nii.gz")
    nib.save(out_nii, out_path)
    print(f"[SUCCESS] Saved 12-Region Master Atlas to {out_path} | Labels: {np.unique(master_atlas)}")

if __name__ == "__main__":
    build_master_12region_atlas()
```

---

### Phase 2 (`src/step2_generate_mri_gt_and_boxes.py`): Cross-Modal MRI Ground Truth & Prompt Extraction
Since APIS NCCT and ADC-MRI scans are already co-registered, registering the MNI template to the high-contrast **ADC-MRI** gives you the silver-standard 3D ground truth ($G$) for evaluation AND the training bounding boxes for 5-fold YOLOv11.

```python
import os
import json
import ants
import numpy as np
import SimpleITK as sitk

def extract_prompts_from_3d_mask(mask_zyx, region_ids=range(1, 13), pad=3):
    """
    Given a 3D label array (Z, H, W), computes for each region k:
    z_min, z_max, z_key (slice with max area), and 2D bounding box [x1, y1, x2, y2].
    """
    prompts = {}
    Z, H, W = mask_zyx.shape
    for rid in region_ids:
        binary_r = (mask_zyx == rid)
        z_indices = np.where(binary_r.any(axis=(1, 2)))[0]
        if len(z_indices) == 0:
            continue
        z_min, z_max = int(z_indices.min()), int(z_indices.max())
        z_key = int(np.argmax(binary_r.sum(axis=(1, 2))))
        
        ys, xs = np.where(binary_r[z_key])
        box = [
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(W - 1, int(xs.max()) + pad),
            min(H - 1, int(ys.max()) + pad)
        ]
        prompts[int(rid)] = {"z_min": z_min, "z_max": z_max, "z_key": z_key, "box": box}
    return prompts

def generate_mri_silver_standard_for_patient(adc_path, ct_path, mni_path, atlas_path, out_gt_path):
    adc_ants = ants.image_read(adc_path)
    ct_ants  = ants.image_read(ct_path)
    mni_ants = ants.image_read(mni_path)
    atl_ants = ants.image_read(atlas_path)

    # Skull-strip MNI template using positive brain mask before registering to skull-stripped APIS ADC
    mni_brain = ants.mask_image(mni_ants, ants.get_mask(mni_ants))
    
    # Non-linear SyN registration: MNI Template -> Patient ADC-MRI
    reg = ants.registration(fixed=adc_ants, moving=mni_brain, type_of_transform='SyN')
    
    # Warp 12-region labels into patient space (resampled onto exact CT grid)
    warped_gt = ants.apply_transforms(
        fixed=ct_ants,
        moving=atl_ants,
        transformlist=reg['fwdtransforms'],
        interpolator='genericLabel'
    )
    ants.image_write(warped_gt, out_gt_path)
    
    # Convert to SimpleITK (Z, H, W) ordering and extract oracle/training boxes
    gt_zyx = sitk.GetArrayFromImage(sitk.ReadImage(out_gt_path)).astype(np.uint8)
    return extract_prompts_from_3d_mask(gt_zyx)
```

---

### Phase 3 (`src/step3_multiwindow_ct_encoder.py`): 3-Channel Multi-Window CT Preprocessor
Converts each raw APIS NCCT scan into the 3-channel `[Stroke_Win, Brain_Win, CSF_Win]` representation for both YOLOv11 and MedSAM2.

```python
import torch
import numpy as np
import SimpleITK as sitk
from PIL import Image

def apply_hu_window(hu_vol, wl, ww):
    low, high = wl - (ww / 2.0), wl + (ww / 2.0)
    clipped = np.clip(hu_vol, low, high)
    return ((clipped - low) / (high - low) * 255.0).astype(np.uint8)

def encode_multiwindow_ct(ct_nifti_path, device="cuda"):
    """
    Returns:
      - multi_win_uint8: (Z, H, W, 3) uint8 array (for saving YOLO slices or visualization)
      - medsam2_tensor : (Z, 3, 512, 512) normalized float32 GPU tensor for MedSAM2
      - ct_sitk        : Original SimpleITK image object (for preserving physical metadata)
    """
    ct_sitk = sitk.ReadImage(ct_nifti_path)
    vol_hu  = sitk.GetArrayFromImage(ct_sitk).astype(np.float32) # (Z, H, W)

    ch_r = apply_hu_window(vol_hu, wl=35, ww=40) # Cortical / Stroke Window (15..55 HU)
    ch_g = apply_hu_window(vol_hu, wl=40, ww=80) # Basal Ganglia Window (0..80 HU)
    ch_b = apply_hu_window(vol_hu, wl=20, ww=40) # CSF / Ventricular Window (0..40 HU)

    multi_win_uint8 = np.stack([ch_r, ch_g, ch_b], axis=-1) # (Z, H, W, 3)
    Z, H, W, _ = multi_win_uint8.shape

    resized = np.zeros((Z, 3, 512, 512), dtype=np.float32)
    for z in range(Z):
        pil_slice = Image.fromarray(multi_win_uint8[z]).resize((512, 512), Image.BILINEAR)
        resized[z] = np.transpose(np.asarray(pil_slice, dtype=np.float32) / 255.0, (2, 0, 1))

    tensor = torch.from_numpy(resized).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    medsam2_tensor = (tensor - mean) / std

    return multi_win_uint8, medsam2_tensor, ct_sitk
```

---

### Phase 4 (`src/step4_prompt_generators.py`): Test-Time Automated Prompt Generators (CT-Only)
At test time, **no MRI is used**. You generate the prompts (`z_min, z_max, z_key, box`) purely from the patient's NCCT scan via either **Mode A (CT-Only 3D Atlas Registration)** or **Mode B (Trained YOLOv11s Detector)**.

```python
import ants
import numpy as np
import SimpleITK as sitk
from ultralytics import YOLO
from step2_generate_mri_gt_and_boxes import extract_prompts_from_3d_mask

# -------------------------------------------------------------------------
# MODE A: CT-Only 3D Deformable Atlas Prompt Generator (Zero Training Needed)
# -------------------------------------------------------------------------
def generate_prompts_modeA_ct_atlas(ct_path, mni_path, atlas_path):
    ct_ants  = ants.image_read(ct_path)
    mni_ants = ants.image_read(mni_path)
    atl_ants = ants.image_read(atlas_path)

    # Window CT to parenchyma range (0..80 HU) so skull/air don't distort registration
    ct_parenchyma = ants.threshold_image(ct_ants, 1, 80) * ct_ants
    mni_brain     = ants.mask_image(mni_ants, ants.get_mask(mni_ants))

    reg = ants.registration(fixed=ct_parenchyma, moving=mni_brain, type_of_transform='Affine')
    warped_atlas = ants.apply_transforms(
        fixed=ct_ants, moving=atl_ants, transformlist=reg['fwdtransforms'], interpolator='genericLabel'
    )
    # Convert ANTs (X, Y, Z) to SimpleITK (Z, H, W)
    atlas_zyx = np.transpose(warped_atlas.numpy(), (2, 1, 0)).astype(np.uint8)
    return extract_prompts_from_3d_mask(atlas_zyx), atlas_zyx

# -------------------------------------------------------------------------
# MODE B: YOLOv11s 3D Prompt Aggregator (Real-Time Inference < 20ms)
# -------------------------------------------------------------------------
def generate_prompts_modeB_yolo(yolo_weights_path, multi_win_uint8, conf_thresh=0.40):
    model = YOLO(yolo_weights_path)
    Z, H, W, _ = multi_win_uint8.shape
    
    # Track detections per class across all Z slices: {rid: [(z, conf, [x1,y1,x2,y2]), ...]}
    detections_per_region = {rid: [] for rid in range(1, 13)}
    
    for z in range(Z):
        res = model.predict(source=multi_win_uint8[z], conf=conf_thresh, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss  = res.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_idx in zip(boxes, confs, clss):
            rid = cls_idx + 1  # YOLO classes 0..11 map to Region IDs 1..12
            detections_per_region[rid].append((z, float(conf), box.astype(int).tolist()))
            
    prompts = {}
    for rid, det_list in detections_per_region.items():
        if not det_list:
            continue
        z_vals = [d[0] for d in det_list]
        # Select the slice with highest YOLO detection confidence as z_key
        best_det = max(det_list, key=lambda d: d[1])
        prompts[rid] = {
            "z_min": int(min(z_vals)),
            "z_max": int(max(z_vals)),
            "z_key": int(best_det[0]),
            "box":   best_det[2]
        }
    return prompts
```

---

### Phase 5 (`src/step5_medsam2_3d_inference.py`): Z-Clamped MedSAM2 & Logit-Argmax Engine
This is the core inference script using `MedSAM2_latest.pt`. It includes toggle flags (`use_z_clamp`, `use_logit_argmax`) so you can run all 5 ablation variants (`A1` through `A5`) from Table 2 with a single function call.

```python
import torch
import numpy as np
import SimpleITK as sitk
from sam2.build_sam import build_sam2_video_predictor_npz

def run_auto_medsam2_3d(
    predictor,
    medsam2_tensor,
    orig_H,
    orig_W,
    region_prompts,
    num_classes=12,
    use_z_clamp=True,
    use_logit_argmax=True
):
    """
    Runs 3D MedSAM2 inference across all prompted regions.
    - use_z_clamp=True      : Restricts tracking to [z_min, z_max] (Ablation A3-A5)
    - use_logit_argmax=True : Resolves overlapping boundaries via Logit Argmax (Ablation A5)
    """
    Z = medsam2_tensor.shape[0]
    
    if use_logit_argmax:
        logit_volume = np.full((num_classes + 1, Z, orig_H, orig_W), -10.0, dtype=np.float32)
        logit_volume[0] = 0.0  # Background threshold at 0.0
    else:
        naive_mask_volume = np.zeros((Z, orig_H, orig_W), dtype=np.uint8)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for rid, p in region_prompts.items():
            z_min = p["z_min"] if use_z_clamp else 0
            z_max = p["z_max"] if use_z_clamp else (Z - 1)
            z_key = p["z_key"]
            box   = np.array(p["box"], dtype=np.float32)

            sub_tensor = medsam2_tensor[z_min : z_max + 1]
            local_key  = z_key - z_min
            inference_state = predictor.init_state(sub_tensor, orig_H, orig_W)

            for reverse_flag in [False, True]:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=local_key,
                    obj_id=1,
                    box=box
                )
                for out_idx, _, logits in predictor.propagate_in_video(inference_state, reverse=reverse_flag):
                    global_z = z_min + out_idx
                    slice_logits = logits[0, 0].cpu().numpy() # Shape: (orig_H, orig_W)
                    
                    if use_logit_argmax:
                        logit_volume[rid, global_z] = np.maximum(logit_volume[rid, global_z], slice_logits)
                    else:
                        naive_mask_volume[global_z, slice_logits > 0.0] = rid

                predictor.reset_state(inference_state)

    if use_logit_argmax:
        return np.argmax(logit_volume, axis=0).astype(np.uint8)
    return naive_mask_volume
```

---

## 5. Execution Checklist & Milestone Schedule

| Milestone | Script / Task | Deliverable Generated | Verification Check |
| :--- | :--- | :--- | :--- |
| **Milestone 1** | Run `step1_build_12region_atlas.py` | `stroke_12regions_atlas.nii.gz` | Open in 3D Slicer / ITK-SNAP; verify labels `1..12` match Left/Right M1, Insula, Caudate, Lentiform, Thalamus, Ventricles. |
| **Milestone 2** | Run `step2_generate_mri_gt_and_boxes.py` on 60 APIS ADC scans | `silver_standard_mri_gt/*.nii.gz` | Overlay generated MRI mask onto APIS `NCCT.nii.gz` to confirm co-registration alignment. |
| **Milestone 3** | Run `Mode A (Atlas-MedSAM2)` across all 60 APIS NCCT scans | `predictions_modeA_atlas_medsam2/*.nii.gz` | First complete 60-patient quantitative table (DSC, IoU, $\text{HD}_{95}$) ready to show your professor. |
| **Milestone 4** | Train 5-Fold `YOLOv11s` on Multi-Window CT slices & run `Mode B (YOLO-MedSAM2)` | `predictions_modeB_yolo_medsam2/*.nii.gz` | Compare runtime (`~30s` Mode A vs. `~1.8s` Mode B) and Dice accuracy in Table 1. |
| **Milestone 5** | Run Ablations `A1..A5` & Stroke Lesion Intersection | Final CSVs, Wilcoxon $p$-values, and 4-panel paper figures | Verify $p < 0.05$ improvement for Multi-Window CT + Z-Clamping + Logit-Argmax. |````

```