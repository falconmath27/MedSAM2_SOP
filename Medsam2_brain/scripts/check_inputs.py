"""Read-only inventory. Uses Python's standard library; does not load scan data."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/project.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolve = lambda key: (ROOT / config[key]).resolve()
    dataset = resolve("dataset_root")
    patients = sorted(p for p in dataset.glob("train_*") if p.is_dir())
    counts = {"ncct": 0, "adc": 0}
    issues = []
    for patient in patients:
        files = [p for p in patient.rglob("*") if p.is_file()
                 and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))]
        for kind in counts:
            matches = [p for p in files if p.name in
                       (f"{patient.name}_{kind}.nii", f"{patient.name}_{kind}.nii.gz")]
            counts[kind] += bool(matches)
            if not matches:
                issues.append(f"{patient.name}: missing {kind}")
            elif len(matches) > 1:
                print(f"REVIEW {patient.name}: {len(matches)} {kind} candidates; select explicitly before use")
    print(f"Dataset: {dataset}")
    print(f"Patients: {len(patients)}/{config['expected_patients']}")
    for kind, count in counts.items():
        print(f"Patients with {kind}: {count}")
    missing_assets = []
    for key in ("medsam2_source", "medsam2_checkpoint", "medsam2_config", "mni_template", "region_atlas"):
        path = resolve(key)
        present = path.is_dir() if key == "medsam2_source" else path.is_file() and path.stat().st_size > 0
        print(f"{'FOUND' if present else 'MISSING'} {key}: {path}")
        if not present:
            missing_assets.append(key)
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for module in ("numpy", "nibabel", "nilearn", "ants", "SimpleITK", "torch", "sam2", "ultralytics"):
        print(f"Module {module}: {'discoverable' if importlib.util.find_spec(module) else 'not installed'}")
    for issue in issues:
        print(f"MISSING {issue}")
    print("Inventory only: file contents, CT HU values, image alignment, model loading and GPU are not validated.")
    return 1 if issues or missing_assets or len(patients) != config["expected_patients"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
