"""Task 2: create a read-only APIS NCCT/ADC inventory and preview overlays.

Run from Medsam2_brain:
    python scripts/1_patient_manifests.py --data-root ./APIS_Dataset

Requires: numpy, nibabel, scipy, matplotlib.
Only JSON and PNG files under data/manifests/ are written.
"""

import argparse
import hashlib
import itertools
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ID_PATTERN = re.compile(r"(?i)(?:^|[^a-z0-9])(train|test|val|case|patient|subject|sub)[_-]?(\d+)(?=$|[^a-z0-9])")
MODALITY_PATTERNS = {
    "ncct": re.compile(r"(?i)(?:^|[_\W])(ncct|non.?contrast|ct)(?=$|[_\W])"),
    "adc": re.compile(r"(?i)(?:^|[_\W])adc(?=$|[_\W])"),
}


def without_nii_suffix(name):
    return re.sub(r"(?i)\.nii(?:\.gz)?$", "", name)


def discover(root):
    """Include nested *directories* ending in .nii; only leaf files are scans."""
    patients = defaultdict(lambda: {"ncct": [], "adc": []})
    ignored = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
            continue
        rel = path.relative_to(root)
        parts = [without_nii_suffix(p) for p in rel.parts]
        # Prefer the file name; consult parent folders for generic filenames.
        modality = None
        for part in [parts[-1], *reversed(parts[:-1])]:
            if re.search(r"(?i)(?:^|[_\W])(mask|label|lesion|segmentation)(?=$|[_\W])", part):
                if part == parts[-1]:
                    modality = "ignore"
                    break
                continue
            matches = [m for m, pattern in MODALITY_PATTERNS.items() if pattern.search(part)]
            if len(matches) == 1:
                modality = matches[0]
                break
        if modality not in ("ncct", "adc"):
            ignored.append(str(path))
            continue

        patient_id = None
        for part in [parts[-1], *reversed(parts[:-1])]:
            match = ID_PATTERN.search(part)
            if match:
                patient_id = f"{match.group(1).lower()}_{int(match.group(2)):03d}"
                break
        if patient_id is None:
            # Fallback: generic filenames (NCCT.nii, ADC.nii) grouped by parent.
            # Keep the relative parent path to avoid merging patients with same folder name.
            parent = rel.parent
            patient_id = str(parent).replace("\\", "/") if parent != Path(".") else ""
            if not patient_id or patient_id.lower() in ("ncct", "ct", "adc"):
                stem = parts[-1]
                patient_id = re.sub(r"(?i)(ncct|adc|ct|non.?contrast)", "", stem).strip("_.- ")
            if not patient_id:
                ignored.append(str(path))
                continue
        patients[patient_id][modality].append(str(path.resolve()))
    return patients, ignored


def image_info(path, modality):
    result = {"path": path, "valid": False, "issues": []}
    try:
        im = nib.load(path)
        if len(im.shape) != 3 or min(im.shape) < 2:
            raise ValueError(f"Expected a 3D volume, got shape {im.shape}")
        a = np.asarray(im.affine, dtype=float)
        if not np.isfinite(a).all() or abs(np.linalg.det(a[:3, :3])) < 1e-10:
            raise ValueError("Invalid or singular spatial affine")
        spacing = [float(v) for v in nib.affines.voxel_sizes(a)]
        if not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise ValueError("Invalid voxel spacing")

        vol = np.asarray(im.dataobj, dtype=np.float32)  # NIfTI slope/intercept applied
        finite = np.isfinite(vol)
        if not finite.all():
            result["issues"].append("Non-finite voxel intensities")
        if not finite.any():
            raise ValueError("Volume contains no finite intensities")
        valid_values = vol[finite]
        lo, med, hi = np.percentile(valid_values, [1, 50, 99])
        vmin, vmax = float(valid_values.min()), float(valid_values.max())
        result.update({
            "shape_xyz": list(im.shape),
            "spacing_mm": [round(v, 5) for v in spacing],
            "orientation": "".join(nib.aff2axcodes(a)),
            "affine": a.tolist(),
            "intensity": {
                "min": vmin, "p01": float(lo), "median": float(med),
                "p99": float(hi), "max": vmax,
            },
        })
        if modality == "ncct":
            # Loose sanity check only; clinical HU calibration needs source verification.
            if (-10 <= vmin <= 10 and -10 <= vmax <= 10) or vmin < -2000 or vmax > 5000:
                result["issues"].append("Suspicious CT intensity range; verify original HU scaling")
            if not np.any((valid_values >= 0) & (valid_values <= 100)):
                result["issues"].append("No voxels in typical brain HU range [0,100]")
        result["valid"] = True
    except Exception as ex:
        result["issues"].append(f"Cannot validate image: {ex}")
    return result


def resolve(candidates, modality):
    """Resolve duplicate scans safely, including exact axis-order reversals.

    The APIS dataset can contain e.g. train_002_ncct.nii(.gz) with LAS
    orientation and train_002_ncct_1.nii.gz with LAI orientation. Their
    raw affines differ even if they represent exactly the same physical image.
    Reorient to canonical RAS *without interpolation*, then compare both
    physical geometry AND voxel values. Never select a mismatch silently.
    """
    infos = [image_info(p, modality) for p in sorted(candidates)]
    good = [x for x in infos if x["valid"]]
    if not good:
        return None, infos, "No readable candidate" if infos else "Missing scan"
    if len(good) != len(infos):
        return None, infos, "Duplicate candidates include unreadable scans; review required"
    if len(good) == 1:
        return good[0], infos, None

    try:
        ref = nib.as_closest_canonical(nib.load(good[0]["path"]))
        ref_values = np.asarray(ref.dataobj, dtype=np.float32)
        for other in good[1:]:
            img = nib.as_closest_canonical(nib.load(other["path"]))
            # Different original orientations are fine if the same world-space
            # points map onto the same canonical voxel indices.
            if img.shape != ref.shape or not np.allclose(
                img.affine, ref.affine, rtol=0, atol=1e-3
            ):
                return None, infos, "Conflicting physical geometries after reorientation; review required"
            values = np.asarray(img.dataobj, dtype=np.float32)
            if not np.array_equal(ref_values, values, equal_nan=True):
                return None, infos, "Duplicate voxel content differs after reorientation; review required"
    except Exception as ex:
        return None, infos, f"Could not compare duplicates: {ex}"

    # All are confirmed exact duplicates in physical space. Prefer the
    # unsuffixed .nii.gz source, but do not modify/delete any other files.
    def preferred(info):
        path = Path(info["path"])
        stem = without_nii_suffix(path.name)
        numbered_copy = bool(re.search(r"_\d+$", stem))
        is_uncompressed = not path.name.lower().endswith(".nii.gz")
        return (numbered_copy, is_uncompressed, path.name.lower())

    return min(good, key=preferred), infos, None


def world_bounds(info):
    dims = np.asarray(info["shape_xyz"], dtype=float) - 1
    corners = np.asarray(list(itertools.product(*[(0, d) for d in dims])))
    coords = nib.affines.apply_affine(np.asarray(info["affine"]), corners)
    return coords.min(axis=0), coords.max(axis=0)


def geometry_check(ct, adc):
    if not ct or not adc:
        return {"status": "not_checked", "reason": "CT or ADC unavailable"}
    ca, cb = world_bounds(ct)
    ma, mb = world_bounds(adc)
    overlap = np.maximum(0, np.minimum(cb, mb) - np.maximum(ca, ma))
    csize, msize = np.maximum(cb - ca, 0), np.maximum(mb - ma, 0)
    denominator = min(np.prod(csize), np.prod(msize))
    ratio = float(np.prod(overlap) / denominator) if denominator > 0 else 0.0
    same_grid = (ct["shape_xyz"] == adc["shape_xyz"] and
                 np.allclose(ct["affine"], adc["affine"], atol=1e-3, rtol=0))
    # Bounding-box overlap is a coarse screening test, NOT a registration measure.
    return {"status": "review_required", "same_voxel_grid": bool(same_grid),
            "physical_fov_overlap_fraction": round(ratio, 4),
            "possible_geometry_problem": bool(ratio < 0.5),
            "note": "Inspect preview; matching geometry does not prove anatomical alignment."}


def save_overlay(patient, out_dir):
    """Header-based resampling for QC only; does not register or modify source files."""
    ct = nib.load(patient["ncct"]["path"])
    adc = nib.load(patient["adc"]["path"])
    adc_on_ct = resample_from_to(adc, (ct.shape, ct.affine), order=1)
    mid = ct.shape[2] // 2
    ct_slice = np.asarray(ct.dataobj[:, :, mid], dtype=float)
    adc_slice = np.asarray(adc_on_ct.dataobj[:, :, mid], dtype=float)
    pos = adc_slice[np.isfinite(adc_slice) & (adc_slice != 0)]
    low, high = np.percentile(pos, [5, 99]) if len(pos) else (0, 1)
    if high <= low:
        high = low + 1
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ct_display = np.clip(ct_slice, 0, 80).T
    axes[0].imshow(ct_display, cmap="gray", origin="lower", vmin=0, vmax=80)
    axes[0].set_title("CT (0–80 HU)")
    axes[1].imshow(adc_slice.T, cmap="gray", origin="lower", vmin=low, vmax=high)
    axes[1].set_title("ADC resampled by header")
    axes[2].imshow(ct_display, cmap="gray", origin="lower", vmin=0, vmax=80)
    axes[2].imshow(np.ma.masked_where(adc_slice.T == 0, adc_slice.T), cmap="magma",
                   origin="lower", vmin=low, vmax=high, alpha=0.40)
    axes[2].set_title("Overlay: inspect alignment")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{patient['patient_id']} | axial voxel slice {mid} | NOT registration proof")
    fig.tight_layout()
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", patient["patient_id"])
    out_path = out_dir / f"{safe_id}_alignment.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "APIS_Dataset")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/manifests/patients.json")
    parser.add_argument("--dev-count", type=int, default=3)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Dataset folder does not exist: {root}")
    groups, ignored = discover(root)
    if not groups:
        parser.error("No recognizable NCCT or ADC .nii/.nii.gz scans found. Check naming/layout.")

    records = []
    for pid, paths in sorted(groups.items()):
        ct, ct_candidates, ct_problem = resolve(paths["ncct"], "ncct")
        adc, adc_candidates, adc_problem = resolve(paths["adc"], "adc")
        issues = []
        for name, issue, selected in [("ncct", ct_problem, ct), ("adc", adc_problem, adc)]:
            if issue:
                issues.append(f"{name}: {issue}")
            if selected:
                issues += [f"{name}: {i}" for i in selected["issues"]]
        geom = geometry_check(ct, adc)
        if geom.get("possible_geometry_problem"):
            issues.append("CT/ADC physical field-of-view overlap is low; examine registration")
        records.append({
            "patient_id": pid, "ncct": ct, "adc": adc,
            "ncct_candidates": ct_candidates, "adc_candidates": adc_candidates,
            "geometry_check": geom, "issues": issues,
            "status": "needs_attention" if issues else "ready_for_visual_review",
            "split": None, "overlay_preview": None,
        })

    paired = [p for p in records if p["ncct"] and p["adc"]]
    # Deterministic patient-level split. No slice from a patient enters both groups.
    ranked = sorted(paired, key=lambda p: hashlib.sha256(("42:" + p["patient_id"]).encode()).hexdigest())
    dev_ids = {p["patient_id"] for p in ranked[:max(args.dev_count, 0)]}
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previews = output.parent / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    for p in records:
        if p["ncct"] and p["adc"]:
            p["split"] = "development" if p["patient_id"] in dev_ids else "evaluation_reserved"
            if p["patient_id"] in dev_ids:
                try:
                    p["overlay_preview"] = save_overlay(p, previews)
                except Exception as ex:
                    p["issues"].append(f"Cannot generate alignment preview: {ex}")
                    p["status"] = "needs_attention"
        else:
            p["split"] = "excluded_until_resolved"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "expected_patients": 60,
        "discovered_patients": len(records),
        "patients_with_both_modalities": len(paired),
        "ignored_nifti_files_count": len(ignored),
        "ignored_nifti_file_examples": ignored[:20],
        "split_seed": 42,
        "notes": ["Splits are provisional until patient IDs and scan duplicates are reviewed.",
                  "Axis-flipped candidates are treated as duplicates only after matching canonical geometry and voxel values.",
                  "No case has verified CT/ADC anatomical alignment automatically.",
                  "Evaluate only after freezing the development/evaluation split."],
        "patients": records,
    }
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Manifest: {output}")
    print(f"Patients found: {len(records)} (expected ~60); paired: {len(paired)}; "
          f"development: {len(dev_ids)}; reserved: {len(paired)-len(dev_ids)}")
    print(f"Needs attention: {sum(bool(p['issues']) for p in records)}")
    print(f"Review CT/ADC previews in: {previews}")
    if len(records) != 60:
        print("WARNING: Patient count differs from the expected 60; inspect file naming and ignored files.")


if __name__ == "__main__":
    main()
