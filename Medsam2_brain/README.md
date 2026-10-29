# 3D Anatomical Brain Segmentation with MedSAM2

Primary working folder: `Medsam2_brain`. Current task-wise plan: [IMPLEMENTATION.md](IMPLEMENTATION.md).
Source reference: [Refer.md](Refer.md). Current scope is 12-region anatomical segmentation;
stroke analysis is excluded.
External files may be referenced; dataset and existing model files remain in their original locations.

## Required inputs

- NCCT brain volumes with original HU intensities, as `.nii` or `.nii.gz`.
- Matching ADC-MRI volumes for reference-mask generation and evaluation. Verify alignment to CT before using them.
- MNI152 template and Harvard-Oxford cortical/subcortical atlases to build the 12-region atlas.
- MedSAM2 code, compatible model configuration, and `MedSAM2_latest.pt` weights.
- Optional later extension: trained 12-class YOLO weights for Mode B. Mode A needs no YOLO training.

After setup, prediction on a new patient requires only their CT scan plus the model/atlas assets.

## Layout

- `configs/project.json`: project-relative paths, CT windows and output labels.
- `scripts/check_inputs.py`: read-only input inventory; detects nested APIS NIfTI files.
- `src/`: pipeline implementation destination.
- `data/master_atlas_3d/`: MNI template and generated 12-region atlas.
- `data/silver_standard_mri_gt/`: generated MRI-derived reference masks.
- `data/prompts_json/`: generated prompts.
- `data/yolo_5fold_dataset/`: future patient-separated training folds.
- `checkpoints/`: future project-specific weights.
- `outputs/`: Mode A/Mode B predictions, ablations, tables and figures.

Existing `data/processed`, `data/aspects_pilot`, and `atlas_downloads` are preserved.
They are excluded from the new pipeline. Obtain the 3D MNI152 and Harvard-Oxford
sources specified in Refer.md and build `anatomy_12regions_atlas.nii.gz` from them.
Raw data is referenced at `../APIS_Dataset`; no duplicate `APIS_raw` copy is needed.

## Start

From this folder, with a working Python interpreter:

```powershell
.venv\Scripts\python.exe scripts/check_inputs.py
```

Exit code 1 indicates missing input assets or an unexpected patient count. Multiple scan
candidates are reported for review; the script never silently chooses a duplicate.
It checks presence only, not image integrity or registration quality.

Task 1 is validated in `.venv` (Python 3.11.14). The imaging imports, ANTs registration,
strict checkpoint loading and three-slice bidirectional CUDA inference passed.
The MedSAM2 source, checkpoint and YAML paths point to the existing sibling repository.

Run validation again with:

```powershell
.venv\Scripts\python.exe scripts/validate_environment.py
```

`environment/validation.json` records hardware, precision checks, checkpoint/config
checksums, source revision and file hashes. `environment/requirements-lock.txt` records
installed versions. `scripts/setup_environment.ps1` recreates the setup using the
primary constraints; use the lock file for exact resolved dependency versions.
Install the CUDA torch packages first; the lock also contains a local MedSAM2 path.

This setup uses the [official PyTorch 2.5.1 / torchvision 0.20.1 CUDA 12.4 pairing](https://docs.pytorch.org/get-started/previous-versions/).
The RTX 3050 A Laptop GPU has 4 GB VRAM and passed FP32/FP16/BF16 matrix checks;
the volume smoke test uses BF16 and CPU state offloading. This does not establish
full-patient memory requirements or anatomical accuracy.

The optional compiled SAM2 connected-components extension is not installed. Set
`predictor.fill_hole_area = 0` after building the predictor: this local builder appends
its own value after caller overrides. Windows attention may fall back to standard
PyTorch kernels. Available system RAM was low during validation; check free RAM
before processing full volumes.

## Implementation checkpoints

1. Validate scan readability, HU values, orientation and CT/MRI alignment on one patient.
2. Build the atlas, check label names/indices, and determine hemisphere using the image affine.
3. Generate and visually review one MRI-derived reference mask.
4. Implement and run Mode A on one patient, then expand to the dataset.
5. Optionally train Mode B with patient-level splits; keep held-out MRI-derived prompts out of inference.
6. Evaluate against the silver standard and report measured results.

The snippets in Refer.md are a starting specification, not a validated pipeline.
Its Mode A text says deformable registration but the code requests Affine; resolve this
before implementation. MRI-derived atlas masks are silver-standard references, not expert
anatomical ground truth. Runtime and statistical-significance statements are targets or
claims to test, not established results. The document does not fully define all ablations
or supply training/evaluation implementations.
