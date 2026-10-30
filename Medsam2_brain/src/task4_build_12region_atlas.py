"""Task 4 — Build and validate the 12-region MNI brain anatomy atlas.

Uses the exact files saved by Task 3. Does not download data or touch CT/MRI.

From Medsam2_brain (Task 1 venv activated):
    python src/task4_build_12region_atlas.py
    python src/task4_build_12region_atlas.py --overwrite   # intentionally rebuild outputs

Requires: numpy, nibabel, matplotlib (installed in the Task 1 environment).

Task 3 inputs: data/atlas_sources_3d/
    mni152_template_1mm.nii.gz
    harvard_oxford_cortical_on_mni152_1mm.nii.gz
    harvard_oxford_subcortical_on_mni152_1mm.nii.gz
    harvard_oxford_cortical_labels.csv
    harvard_oxford_subcortical_labels.csv
    atlas_source_metadata.json

Task 4 outputs: data/master_atlas_3d/
    mni_template_1mm.nii.gz
    anatomy_12regions_atlas.nii.gz
    atlas_label_counts.csv
    atlas_build_metadata.json
    previews/anatomy_12regions_overview_3planes.png
    previews/region_01_3planes.png ... region_12_3planes.png

Label 0 means "not among selected regions", NOT only nonbrain background.
All anatomical left/right decisions use RAS+ WORLD x from the image affine:
negative x = anatomical left, positive x = anatomical right.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "data" / "atlas_sources_3d"
DEFAULT_OUTPUT = ROOT / "data" / "master_atlas_3d"

# Final label order, per IMPLEMENTATION.md. Every pair is Left, Right.
REGIONS = {
    1: "Left precentral gyrus",
    2: "Right precentral gyrus",
    3: "Left insular cortex",
    4: "Right insular cortex",
    5: "Left caudate nucleus",
    6: "Right caudate nucleus",
    7: "Left lentiform nucleus (putamen + pallidum)",
    8: "Right lentiform nucleus (putamen + pallidum)",
    9: "Left thalamus",
    10: "Right thalamus",
    11: "Left lateral ventricle",
    12: "Right lateral ventricle",
}

# Names rather than numerical source IDs are the authoritative mapping. Numerical
# IDs in Refer.md are only expectations and may differ between atlas releases.
CORTICAL_TARGETS = [
    ("Precentral Gyrus", 1, 2),
    ("Insular Cortex", 3, 4),
]
SUBCORTICAL_TARGETS = [
    (5, ("Left Caudate",)),
    (6, ("Right Caudate",)),
    (7, ("Left Putamen", "Left Pallidum")),
    (8, ("Right Putamen", "Right Pallidum")),
    (9, ("Left Thalamus",)),
    (10, ("Right Thalamus",)),
    (11, ("Left Lateral Ventricle",)),
    (12, ("Right Lateral Ventricle",)),
]
REFERENCE_IDS = {
    "cortical": {"Precentral Gyrus": 7, "Insular Cortex": 2},
    "subcortical": {
        "Left Caudate": 5, "Right Caudate": 16,
        "Left Putamen": 6, "Left Pallidum": 7,
        "Right Putamen": 17, "Right Pallidum": 18,
        "Left Thalamus": 4, "Right Thalamus": 15,
        "Left Lateral Ventricle": 3, "Right Lateral Ventricle": 14,
    },
}
COLORS = [
    "#de5366", "#fa93a1", "#ef9e3f", "#fbd287",
    "#4aa68b", "#91d5b9", "#5584d6", "#a4bdeb",
    "#8b68c7", "#bda5e6", "#388fba", "#8fd1e6",
]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(name: str) -> str:
    """Handle whitespace/punctuation and an old atlas spelling: 'Ventrical'."""
    name = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    return re.sub(r"\bventrical\b", "ventricle", name)


def load_lookup(csv_path: Path) -> dict[str, int]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"index", "name"}.issubset(reader.fieldnames):
            raise ValueError(f"Invalid Task 3 label table: {csv_path}")
        lookup = {}
        found_ids = set()
        for row in reader:
            key, idx = normalize(row["name"]), int(row["index"])
            if key in lookup or idx in found_ids:
                raise ValueError(f"Repeated/ambiguous source label in {csv_path.name}: {row}")
            lookup[key] = idx
            found_ids.add(idx)
    if not lookup:
        raise ValueError(f"Empty atlas label table: {csv_path}")
    return lookup


def resolve(lookup: dict[str, int], source_name: str, atlas_type: str,
            warnings: list[str]) -> int:
    key = normalize(source_name)
    if key not in lookup:
        raise ValueError(
            f"Cannot find '{source_name}' in {atlas_type} atlas labels. "
            f"Inspect data/atlas_sources_3d/harvard_oxford_{atlas_type}_labels.csv"
        )
    actual = lookup[key]
    expected = REFERENCE_IDS[atlas_type].get(source_name)
    if expected is not None and actual != expected:
        warnings.append(
            f"{atlas_type}: '{source_name}' is source ID {actual} "
            f"(Refer.md expected {expected}); using verified Task 3 table"
        )
    return actual


def require_3d(img, name: str, *, atlas: bool = False) -> np.ndarray:
    if len(img.shape) != 3:
        raise ValueError(f"{name}: expected a 3D NIfTI volume, got {img.shape}")
    if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-8:
        raise ValueError(f"{name}: invalid NIfTI affine")
    array = np.asarray(img.dataobj)
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: nonfinite voxels")
    if atlas and (not np.array_equal(array, np.rint(array)) or array.min() < 0):
        raise ValueError(f"{name}: labels are negative or non-integer")
    return array


def load_sources(source_dir: Path):
    names = {
        "mni_template": "mni152_template_1mm.nii.gz",
        "cortical_on_mni": "harvard_oxford_cortical_on_mni152_1mm.nii.gz",
        "subcortical_on_mni": "harvard_oxford_subcortical_on_mni152_1mm.nii.gz",
        "cortical_labels": "harvard_oxford_cortical_labels.csv",
        "subcortical_labels": "harvard_oxford_subcortical_labels.csv",
    }
    paths = {key: source_dir / filename for key, filename in names.items()}
    meta_path = source_dir / "atlas_source_metadata.json"
    for path in [*paths.values(), meta_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing Task 3 output: {path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for name, expected_variant in (
        ("cortical", "cort-maxprob-thr25-1mm"),
        ("subcortical", "sub-maxprob-thr25-1mm"),
    ):
        recorded = meta.get("harvard_oxford", {}).get(name, {})
        if recorded.get("requested_variant") != expected_variant:
            raise ValueError(f"Task 3 {name} atlas must be {expected_variant}")
        if recorded.get("resample_interpolation") != "nearest":
            raise ValueError(f"Task 3 {name} atlas was not resampled by nearest-neighbor")
        if recorded.get("symmetric_split") is not False:
            raise ValueError(f"Unexpected {name} symmetric_split; expected False")

    for key, path in paths.items():
        recorded_hash = meta.get("sha256", {}).get(key)
        if not recorded_hash:
            raise ValueError(f"Task 3 metadata missing checksum for {path.name}")
        if checksum(path) != recorded_hash:
            raise ValueError(f"Task 3 input changed since creation: {path.name}")

    template = nib.load(str(paths["mni_template"]))
    cort = nib.load(str(paths["cortical_on_mni"]))
    sub = nib.load(str(paths["subcortical_on_mni"]))
    intensity = require_3d(template, "MNI152 template")
    cortical = require_3d(cort, "Cortical source", atlas=True).astype(np.int16)
    subcortical = require_3d(sub, "Subcortical source", atlas=True).astype(np.int16)
    spacing = nib.affines.voxel_sizes(template.affine)
    if not np.allclose(spacing, [1, 1, 1], atol=1e-3, rtol=0):
        raise ValueError(f"MNI template is not 1 mm isotropic: {spacing}")
    for label_img, label_name in [(cort, "Cortical"), (sub, "Subcortical")]:
        if label_img.shape != template.shape or not np.allclose(
            label_img.affine, template.affine, atol=1e-4, rtol=0
        ):
            raise ValueError(f"{label_name} atlas is not on the exact MNI template grid")
    return template, intensity, cortical, subcortical, paths


def world_x(coords: tuple[np.ndarray, np.ndarray, np.ndarray], affine: np.ndarray) -> np.ndarray:
    """Calculate RAS+ X coordinate without assuming anatomical left is index 0."""
    return (affine[0, 0] * coords[0] + affine[0, 1] * coords[1]
            + affine[0, 2] * coords[2] + affine[0, 3])


def compose_atlas(template, cortical: np.ndarray, subcortical: np.ndarray,
                  cort_lut: dict[str, int], sub_lut: dict[str, int]):
    dest = np.zeros(template.shape, dtype=np.uint8)
    affine = np.asarray(template.affine, dtype=np.float64)
    warnings = []
    assignments = []
    overlap_voxels = {}

    # Cortical Harvard–Oxford atlas is bilateral. Split by the MNI mid-sagittal
    # plane X=0 mm, not by half of the voxel array.
    for source_name, left_dest, right_dest in CORTICAL_TARGETS:
        idx = resolve(cort_lut, source_name, "cortical", warnings)
        coords = np.where(cortical == idx)
        if coords[0].size == 0:
            raise ValueError(f"Empty cortical source structure: {source_name} (ID {idx})")
        x = world_x(coords, affine)
        for final_id, selected in ((left_dest, x < 0), (right_dest, x > 0)):
            target_voxels = tuple(axis[selected] for axis in coords)
            if len(target_voxels[0]) == 0:
                raise ValueError(f"No voxels for {REGIONS[final_id]}")
            dest[target_voxels] = final_id
            assignments.append({"output_id": final_id, "source_atlas": "cortical",
                                "source_names": [source_name], "source_ids": [idx]})
        # voxels on exactly x = 0 mm remain 0 and are counted in metadata
        on_midline = int(np.count_nonzero(x == 0))
        if on_midline:
            warnings.append(f"{source_name}: {on_midline} cortical voxels on world x=0 left unassigned")

    # Subcortical atlas already has left/right entries. It takes priority over
    # cortical voxels where the two Harvard–Oxford atlases overlap.
    for final_id, source_names in SUBCORTICAL_TARGETS:
        ids = [resolve(sub_lut, name, "subcortical", warnings) for name in source_names]
        for idx, name in zip(ids, source_names):
            if not np.any(subcortical == idx):
                raise ValueError(f"Subcortical source {name} (ID {idx}) has no voxels")
        mask = np.isin(subcortical, ids)
        overlap_voxels[str(final_id)] = int(np.count_nonzero(mask & (dest != 0)))
        dest[mask] = final_id
        assignments.append({"output_id": final_id, "source_atlas": "subcortical",
                            "source_names": list(source_names), "source_ids": ids})

    present, counts = np.unique(dest, return_counts=True)
    if set(present.tolist()) != set(range(13)):
        raise ValueError(f"Expected labels 0–12 all present, but got {present.tolist()}")

    volume_per_voxel = abs(float(np.linalg.det(affine[:3, :3])))
    summary = []
    for final_id in range(13):
        if final_id == 0:
            summary.append({"label_id": 0, "anatomy": "Other tissue / outside selected regions",
                            "voxel_count": int(counts[present.tolist().index(0)]),
                            "volume_mm3": round(int(counts[present.tolist().index(0)]) * volume_per_voxel, 1),
                            "centroid_x_mm": ""})
            continue
        coords = np.where(dest == final_id)
        x = world_x(coords, affine)
        center = float(x.mean())
        expected_left = final_id % 2 == 1
        if (expected_left and center >= 0) or (not expected_left and center <= 0):
            raise ValueError(f"{REGIONS[final_id]} on wrong side: world-x centroid={center:.2f} mm")
        wrong_side = float(np.mean(x > 0 if expected_left else x < 0))
        if wrong_side > 0.05:
            warnings.append(f"{REGIONS[final_id]}: {wrong_side:.1%} voxels across midline; inspect")
        summary.append({"label_id": final_id, "anatomy": REGIONS[final_id],
                        "voxel_count": len(x),
                        "volume_mm3": round(len(x) * volume_per_voxel, 1),
                        "centroid_x_mm": round(center, 3)})
    return dest, summary, assignments, overlap_voxels, warnings


def write_preview(template_img, label_img, output: Path, region_id: int | None = None,
                  canonical_arrays=None):
    if canonical_arrays is None:
        tm = nib.as_closest_canonical(template_img)
        lm = nib.as_closest_canonical(label_img)
        template = np.asarray(tm.dataobj)
        labels = np.asarray(lm.dataobj)
    else:
        template, labels = canonical_arrays
    mask = labels > 0 if region_id is None else labels == region_id
    if not mask.any():
        raise ValueError(f"No voxels for preview label {region_id}")

    x = int(np.argmax(np.count_nonzero(mask, axis=(1, 2))))
    y = int(np.argmax(np.count_nonzero(mask, axis=(0, 2))))
    z = int(np.argmax(np.count_nonzero(mask, axis=(0, 1))))
    views = [
        ("Axial", template[:, :, z].T, labels[:, :, z].T),
        ("Coronal", template[:, y, :].T, labels[:, y, :].T),
        ("Sagittal", template[x, :, :].T, labels[x, :, :].T),
    ]

    cmap = ListedColormap(["#000000", *COLORS])
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(np.arange(-0.5, 13.5, 1), cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    for ax, (view_name, bg, m) in zip(axes, views):
        nonzero_bg = bg[bg > 0]
        low, high = np.percentile(nonzero_bg, [1, 99]) if len(nonzero_bg) else (0, 1)
        if high <= low:
            high = low + 1
        ax.imshow(bg, cmap="gray", origin="lower", vmin=low, vmax=high)
        masked = np.ma.masked_equal(m if region_id is None else np.where(m == region_id, m, 0), 0)
        ax.imshow(masked, cmap=cmap, norm=norm, origin="lower", interpolation="nearest", alpha=0.65)
        ax.set_title(view_name)
        ax.set_axis_off()
    if region_id is None:
        fig.suptitle("MNI152: combined 12-region anatomical atlas", fontsize=14)
        patches = [Patch(facecolor=COLORS[i - 1], label=f"{i}: {REGIONS[i]}") for i in range(1, 13)]
        fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
                   bbox_to_anchor=(0.5, -0.005))
        fig.tight_layout(rect=(0, 0.19, 1, 0.92))
    else:
        fig.suptitle(f"Region {region_id}: {REGIONS[region_id]}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("Task 4 output must not overwrite Task 3 sources")
    print("[1/4] Reading and validating Task 3 MNI152/Harvard–Oxford files...")
    template, _, cortical, subcortical, paths = load_sources(source_dir)
    cort_lut = load_lookup(paths["cortical_labels"])
    sub_lut = load_lookup(paths["subcortical_labels"])

    print("[2/4] Building all 12 anatomical labels (world-space hemisphere checks)...")
    labelmap, rows, mapping, overlap, warnings = compose_atlas(
        template, cortical, subcortical, cort_lut, sub_lut
    )
    previews = output_dir / "previews"
    out_paths = [
        output_dir / "mni_template_1mm.nii.gz",
        output_dir / "anatomy_12regions_atlas.nii.gz",
        output_dir / "atlas_label_counts.csv",
        output_dir / "atlas_build_metadata.json",
        previews / "anatomy_12regions_overview_3planes.png",
        *(previews / f"region_{i:02d}_3planes.png" for i in range(1, 13)),
    ]
    existing = [p for p in out_paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Task 4 outputs already exist ({existing[0]}). "
            "Use --overwrite only if you intend to rebuild them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    template_path, atlas_path, csv_path, json_path = out_paths[:4]

    print("[3/4] Saving master atlas with original MNI NIfTI affine and 1 mm spacing...")
    shutil.copy2(paths["mni_template"], template_path)
    header = template.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1.0, 0.0)
    final_img = nib.Nifti1Image(labelmap, template.affine.copy(), header=header)
    nib.save(final_img, str(atlas_path))
    stored = nib.load(str(atlas_path))
    if stored.shape != template.shape or not np.allclose(stored.affine, template.affine, atol=1e-4, rtol=0):
        raise ValueError("Saved final atlas geometry differs from the MNI template")
    if not np.array_equal(np.asarray(stored.dataobj), labelmap):
        raise ValueError("Saved atlas voxel data does not match the generated label map")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("[4/4] Creating anatomical QC previews...")
    canon_t = np.asarray(nib.as_closest_canonical(template).dataobj, dtype=np.float32)
    canon_labels = np.asarray(nib.as_closest_canonical(stored).dataobj, dtype=np.uint8)
    canonical = (canon_t, canon_labels)
    write_preview(template, stored, out_paths[4], canonical_arrays=canonical)
    for label_id, output in enumerate(out_paths[5:], start=1):
        write_preview(template, stored, output, region_id=label_id, canonical_arrays=canonical)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Task 4 — 12-region anatomical atlas",
        "source_dir": str(source_dir),
        "source_sha256": {key: checksum(value) for key, value in paths.items()},
        "source_name_to_label_id_mapping": mapping,
        "anatomical_hemisphere_definition": "RAS world X < 0 mm = left; > 0 mm = right",
        "overlap_policy": "subcortical atlas overwrites cortical atlas",
        "subcortical_overlap_counts": overlap,
        "label_0": "Everything not among the 12 selected regions, including other brain tissue",
        "region_counts": rows,
        "warnings": warnings,
        "output_sha256": {"atlas": checksum(atlas_path), "template": checksum(template_path)},
        "shape_xyz": list(stored.shape),
        "affine": stored.affine.tolist(),
        "important_note": "Visual QC is still required. This atlas is NOT patient-space segmentation.",
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[SUCCESS] Task 4 completed")
    print(f"Master atlas: {atlas_path}")
    print(f"Template:     {template_path}")
    print(f"Counts CSV:   {csv_path}")
    print(f"Previews:     {previews}")
    print(f"Labels:       {np.unique(labelmap).tolist()}")
    for row in rows[1:]:
        print(f"  {row['label_id']:2d}  {row['anatomy']:<49} {row['voxel_count']:>9,} voxels")
    for warning in warnings:
        print(f"[REVIEW] {warning}")
    print("Inspect overview and all region previews before beginning Task 5.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4: Build 12-region anatomical MNI atlas")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCES,
                        help="Task 3 atlas source folder")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination for master atlas")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing Task 4 outputs")
    args = parser.parse_args()
    run(args.source_dir, args.output_dir, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"[ERROR] {type(exc).__name__}: {exc}") from exc
