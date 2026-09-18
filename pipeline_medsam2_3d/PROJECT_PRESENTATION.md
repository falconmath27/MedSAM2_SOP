# Slide 1: Project Intro

## 3D Knee CT Bone Segmentation with MedSAM2 (Multi-View)

- Goal: segment knee bone structures across full 3D CT volumes.
- Approach: combine YOLO-based localization + MedSAM2 video propagation.
- Views used: axial, sagittal, and coronal.
- Final outputs: per-view masks, fused masks, and NIfTI volumes for evaluation.

---

# Slide 2: Problem Statement

- Accurate 3D bone segmentation from CT is hard with slice-by-slice methods.
- Key challenges:
  - discontinuous detections across slices,
  - orientation/alignment mismatches across views,
  - GPU memory limits for long 3D volumes (~700 slices),
  - need for robust NIfTI geometry consistency.
- Requirement: generate stable, high-quality volumetric masks that can be compared to ground truth.

---

# Slide 3: Methodology (Inference Pipeline)

1. Data preparation: each CT volume is exported into ordered PNG slices for axial, sagittal, and coronal views.
2. Detection stage: batched YOLO predicts bone boxes per slice, then low-confidence and duplicate-nearby boxes are filtered.
3. Side assignment: detections are split into left/right groups using the slice midline.
4. 3D tracking: boxes are linked across adjacent slices using IoU and gap tolerance to form persistent bone tracks.
5. Prompt selection: key frames are chosen per track (largest visible bone area), plus periodic prompt frames to maintain coverage.
6. Segmentation core: MedSAM2 runs in video mode on overlapping chunks, propagating masks forward/backward for each tracked bone.

---

# Slide 4: Methodology (Reconstruction, Fusion, Evaluation)

- Chunk management: long volumes are processed in overlapping windows to fit GPU memory while preserving continuity.
- Leakage control: negative prompt points from neighboring tracks are used to reduce cross-bone mask spillover.
- Output reconstruction: per-track masks are merged per slice, then converted from 2D masks to 3D NIfTI with DICOM-based geometry/orientation.
- Multi-view fusion: final masks are generated using strict intersection (AND) and majority vote (2-of-3) across views.
- Post-processing: optional 3D Gaussian smoothing and thresholding are applied for cleaner volumetric masks.
- Evaluation protocol: IoU is reported overall, per side (left/right), and by fixed slice groups (for example, 50-slice bins).

---

# Slide 5: Results (Patient 15 Example)

- Produced directional and fused NIfTI outputs for multiple patients (p4, p5, p6, p15, p16, p17, p18).
- For smooth axial+sagittal intersection vs ground truth (group size = 50):
  - Overall IoU: 0.5963
  - Right IoU: 0.8683
  - Left IoU: 0.0000
- Slices 400-600 combined IoU:
  - 0.9076 (computed as total intersection / total union, not simple average).
- Observation: strongest overlap appears in mid-late slice ranges (groups 9-12).

---

# Slide 6: Conclusion and Next Steps

- The pipeline reliably generates 3D masks with scalable chunked inference.
- Multi-view fusion improves robustness in high-confidence regions.
- Main remaining issue: side imbalance/alignment error (left side underperforming in p15).

Next steps:

- improve side-aware alignment and orientation validation,
- refine prompting/tracking around sparse regions,
- add patient-wise QA dashboard with per-region IoU trends.
