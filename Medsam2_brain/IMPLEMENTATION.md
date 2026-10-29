# 3D Anatomical Brain Segmentation with MedSAM2

Updated: 2026-10-30

## Objective and scope

Build a pipeline that takes a patient's 3D non-contrast CT volume and produces a
3D anatomical label map using MedSAM2, with automatic prompts from a 3D atlas.

The user's current scope overrides the stroke-related portions of `Refer.md`:

- Segment anatomy only. No stroke detection, lesion-mask inputs, infarct burden,
  stroke localization, or stroke F1 evaluation.
- Use the MNI152 1 mm template and Harvard-Oxford cortical/subcortical **3D volumes**
  described in `Refer.md`.
- Exclude the existing `atlas_downloads`, ASPECTS pilot and earlier processed data
  from this pipeline. Preserve these files; do not use them as reference anatomy.
- Start with the 12 anatomical regions specified in `Refer.md`. This is selected-region
  parcellation, not a complete labeling of every structure in the brain.
- MedSAM2 remains pretrained and inference-only initially. YOLO training is a later
  extension from `Refer.md`, not a prerequisite for completing the atlas-based pipeline.
- ADC-MRI is for generating evaluation references, not an input to CT-only prediction.

## Inputs and outputs

| Item | Use |
| --- | --- |
| Patient NCCT `.nii` / `.nii.gz` | Required prediction input; preserve HU values and geometry |
| Paired ADC-MRI | Generate MRI-derived reference labels for evaluation |
| MNI152 1 mm template | Spatial registration reference |
| Harvard-Oxford `cort-maxprob-thr25-1mm` | Source cortical labels |
| Harvard-Oxford `sub-maxprob-thr25-1mm` | Source subcortical labels |
| MedSAM2 code, checkpoint and matching configuration | Segmentation engine |
| Output label volume | Integer labels 0–12 on the original CT grid |
| Output previews and metrics | Visual inspection and quantitative evaluation |

Raw scans currently live in `../APIS_Dataset`; existing MedSAM2 assets are referenced
from `../pipeline_medsam2/MedSAM2`. Avoid copying these large inputs unnecessarily.

## Target labels

| Left label | Right label | Anatomical structure |
| --- | --- | --- |
| 1 | 2 | Precentral gyrus |
| 3 | 4 | Insular cortex |
| 5 | 6 | Caudate nucleus |
| 7 | 8 | Lentiform nucleus: putamen + pallidum |
| 9 | 10 | Thalamus |
| 11 | 12 | Lateral ventricle |

Label 0 means everything outside the selected regions, including other brain tissue.
Use anatomical names rather than stroke-territory abbreviations in code and reports.

## Task 1 — Prepare and verify the environment

- [x] Create an isolated Python environment inside `Medsam2_brain` with versions
  compatible with the local MedSAM2 source and registration packages.
- [x] Install the imaging dependencies from `requirements.txt`; install matching
  PyTorch/torchvision and the local MedSAM2 package.
- [x] Check GPU availability, available memory and supported precision.
- [x] Load the configured checkpoint successfully with its matching YAML.
- [x] Record package versions, model checksum and source revision after validation.

**Validated:** Python 3.11.14, CUDA PyTorch 2.5.1 / torchvision 0.20.1,
ANTs registration and three-slice bidirectional MedSAM2 inference passed.
See `environment/validation.json` and `environment/requirements-lock.txt` for evidence.
Optional connected-components postprocessing is disabled; attention may use fallback kernels.

## Task 2 — Build a clean patient manifest

Planned file: `scripts/build_patient_manifest.py`.

- [ ] Discover NCCT and ADC files in the actual APIS directory structure, including
  nested folders whose names end in `.nii`.
- [ ] Resolve duplicate compressed/uncompressed candidates by checking their contents
  and geometry; record one explicit path per modality and patient.
- [ ] Record patient ID, paths, dimensions, spacing, orientation and modality availability.
- [ ] Check volume readability, finite intensities and plausible CT HU scaling.
- [ ] Check CT/ADC alignment using physical geometry and visual overlays. Matching
  dimensions alone do not establish alignment; register modalities if necessary.
- [ ] Select a small development set for visual debugging; reserve other patients
  for evaluation before tuning parameters.

**Deliverable:** `data/manifests/patients.json`, with missing or unsuitable cases flagged.
**Done when:** every selected case has unambiguous inputs and documented geometry.
No lesion-mask paths are needed.

## Task 3 — Obtain the specified 3D atlas sources

Planned file: `src/step1_build_12region_atlas.py`.

- [ ] Load/download MNI152 at 1 mm and fetch the two Harvard-Oxford variants above
  using Nilearn, as specified in `Refer.md`.
- [ ] Store new source assets in `data/atlas_sources_3d/` and keep them separate from
  the existing atlas folders.
- [ ] Record exact atlas variants, source information, label tables and checksums.
- [ ] Verify each source is a readable 3D volume with valid physical coordinates.
- [ ] Resample atlas labels to the template grid using nearest-neighbor interpolation.

**Done when:** template and label volumes share the intended physical grid and can
be viewed together in axial, coronal and sagittal planes.

## Task 4 — Build and inspect the 12-region atlas

Continue in `src/step1_build_12region_atlas.py`.

- [ ] Match region names against the downloaded label tables. Verify the numerical
  indices in `Refer.md` instead of assuming they are correct for every downloaded variant.
- [ ] Separate left/right cortical anatomy using world coordinates from the affine;
  do not assume array index direction establishes anatomical left and right.
- [ ] Combine putamen and pallidum into the lentiform label on each side.
- [ ] Define deterministic overlap handling and produce integer labels 0–12.
- [ ] Save `data/master_atlas_3d/mni_template_1mm.nii.gz` and
  `data/master_atlas_3d/anatomy_12regions_atlas.nii.gz`.
- [ ] Inspect all 12 structures and their left/right assignments in three planes.

**Done when:** all target regions are present, labels are valid, and the saved geometry
matches the template. Save preview images and per-region voxel counts.

## Task 5 — Generate MRI-derived evaluation references

Planned file: `src/step2_generate_mri_references.py`.

- [ ] Register the MNI template to the patient's ADC-MRI, starting with one development case.
- [ ] Apply the transform to the 12-region label atlas with label-preserving interpolation.
- [ ] If ADC and CT are not aligned, compose the verified ADC-to-CT transform before
  placing reference labels on the original CT grid.
- [ ] Save transforms and reference masks in `data/silver_standard_mri_gt/`.
- [ ] Inspect overlays for each region; flag failed registrations instead of treating
  them as valid reference anatomy.

**Done when:** a reviewed reference map exists on the CT grid for each evaluation case.
These are **silver-standard** atlas-derived references, not expert anatomical ground truth.
They share an atlas source with prompting, so acknowledge this dependency in evaluation.

## Task 6 — Prepare CT volumes for MedSAM2

Planned file: `src/step3_multiwindow_ct_encoder.py`.

- [ ] Define consistent array conventions: image physical coordinates, NIfTI voxel
  indices and model slice order must have explicit conversions.
- [ ] Implement the three windows from `Refer.md`: level/width 35/40, 40/80 and 20/40.
  Name the first channel `narrow_brain`; its use does not imply a stroke objective.
- [ ] Produce a three-channel slice stack in a documented channel order.
- [ ] Verify resize dimensions and normalization against the actual local MedSAM2
  predictor; do not normalize twice or assume the snippet matches every model version.
- [ ] Retain original CT geometry and the mapping between original pixels and model input.

**Done when:** a CT volume becomes a valid model input and coordinates can be mapped
back to the original volume without swapping axes or reversing anatomy.

## Task 7 — Generate automatic anatomical prompts from CT

Planned file: `src/step4_prompt_generators.py`.

- [ ] Register the MNI template to CT using an appropriate intensity preprocessing
  and multimodal registration configuration, then transform the atlas to the CT grid.
- [ ] Start with the affine approach present in the reference code. The document also
  calls this deformable registration; evaluate a nonlinear refinement separately if
  affine alignment is inadequate and record which method is used.
- [ ] For each region, extract `z_min`, `z_max`, the largest-area `z_key`, and a padded
  bounding box on that slice. Clamp boxes and slice limits to valid image bounds.
- [ ] Save prompts, transformed atlas, registration settings and quality flags.
- [ ] Inspect prompts over CT and handle absent or failed regions explicitly.

**Done when:** prompts for a development case are anatomically plausible and generated
from CT plus atlas assets only. MRI-derived labels must not supply production prompts.

## Task 8 — Implement 3D MedSAM2 segmentation

Planned file: `src/step5_medsam2_3d_inference.py`.

- [ ] Validate the local predictor's input format and prompt-coordinate conventions.
- [ ] Treat the ordered CT slice stack as the volume sequence for MedSAM2 propagation.
- [ ] For each region, restrict propagation to its configured slice range and prompt
  the key slice with its bounding box.
- [ ] Propagate forward and backward from the key slice with correct state management.
- [ ] Map returned logits back to original slice indices and CT pixel dimensions.
- [ ] Merge region logits using maximum positive foreground logit; assign background
  when no region exceeds the threshold. Define deterministic tie handling.
- [ ] Use memory-conscious accumulation rather than requiring all full-resolution
  region-logit volumes to remain on the GPU at once.
- [ ] Save the final integer volume with original CT affine, spacing and dimensions.

**Done when:** one complete 3D CT produces a valid anatomical label map. Inspect
cross-slice continuity, left/right consistency, missing regions and overlapping boundaries.

## Task 9 — Add a reproducible single-case runner

Planned file: `scripts/run_case.py`.

- [ ] Connect preprocessing, CT registration, prompting, MedSAM2 and output writing.
- [ ] Accept an explicit patient/CT path and configuration; make MRI optional and
  evaluation-only.
- [ ] Write per-run parameters, input/model identifiers, timing, failures and outputs.
- [ ] Save overlays and a simple completion summary under a unique run directory.
- [ ] Refuse accidental output overwrites unless explicitly requested.

**Done when:** the same case can be run from one documented command and its output
can be traced to its inputs, configuration and model.

## Task 10 — Validate anatomy and measure segmentation quality

Planned file: `src/step6_evaluate_anatomy.py`.

- [ ] Compare predictions with reviewed MRI-derived references on the same CT grid.
- [ ] Compute Dice, IoU and HD95 in millimeters per region and patient. Define handling
  of empty predictions/references and report failures rather than silently dropping them.
- [ ] Compare MedSAM2 results against the CT-warped atlas alone as a baseline.
- [ ] Review three-plane overlays and report which structures consistently fail.
- [ ] Measure runtime and peak memory. Report observed results without assuming
  the runtime or significance claims in `Refer.md` will hold.
- [ ] If comparison experiments are useful, define them explicitly: single versus
  three-window encoding, propagation with/without slice limits, and overlap-merging choices.

**Done when:** reviewed figures and per-case/per-region metric tables are available,
with reference-quality limitations documented. No stroke-specific metrics are produced.

## Task 11 — Run the full dataset and finalize the core project

Planned file: `scripts/run_batch.py`.

- [ ] Run the small development set first and resolve technical failures.
- [ ] Freeze configuration, then process the remaining eligible cases.
- [ ] Support resuming completed runs, logging failures and retrying failed cases.
- [ ] Account for every discovered case, including exclusions and their reasons.
- [ ] Save final results under `outputs/predictions_modeA_atlas_medsam2/` and
  `outputs/tables_and_figures/`.
- [ ] Document installation, single-case execution, batch execution and evaluation in README.

**Core project complete when:** a reproducible CT-only MedSAM2 pipeline creates 3D
12-region label maps, preserves patient geometry, has reviewed anatomical overlays,
and has transparent quantitative evaluation across eligible cases. Completion does not
imply that a particular accuracy target has been achieved; choose acceptance thresholds
before final evaluation, informed by the development results.

## Task 12 — Optional later extension: YOLO anatomical prompting

Only pursue this after the atlas-based pipeline works and this extension is selected.

- [ ] Generate anatomical training boxes from reviewed training-patient reference masks.
- [ ] Create patient-level folds; never split slices from one patient across train/test.
- [ ] Train a detector for the same 12 anatomy labels.
- [ ] Aggregate detections across slices into key-slice boxes and slice ranges.
- [ ] Feed these prompts into the same MedSAM2 inference and evaluation components.
- [ ] Compare speed, anatomical accuracy and failures against atlas prompting on held-out patients.

No MedSAM2 fine-tuning is required by the initial plan.

## Current starting point

- Folder scaffold, path configuration, dependency inventory and read-only input checker exist.
- Previous inventory found CT and ADC candidates for 60 cases and local MedSAM2 assets;
  duplicate inputs still require resolution and image contents remain unvalidated.
- Task 1 environment validation is complete. The new 3D atlas and pipeline implementation remain outstanding.
- `Refer.md` remains unchanged as source material; this document records the current scope.
- Execute Tasks 1–4 first, then validate Tasks 5–10 on one case before scaling up.
