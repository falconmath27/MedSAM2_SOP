"""Task 3: fetch, validate, and align 3D reference atlas *sources*.

Run from Medsam2_brain (with Task 1 environment active):
    python src/step1_build_12region_atlas.py

Dependencies: numpy, nibabel, nilearn, matplotlib (and Nilearn dependencies).
Requires internet access on first run for the Harvard-Oxford atlases.

Writes exclusively to data/atlas_sources_3d/ by default:
  - MNI152 1 mm anatomical template
  - Original Harvard-Oxford cortical/subcortical 3D atlases
  - Label-preserving, nearest-neighbor copies on the MNI grid
  - Indexed label tables, source metadata, SHA-256 checksums, and 3-plane PNGs

Task 4 will use these validated sources to build the FINAL 12-region atlas.
No segmentation labels 1..12 are constructed in Task 3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets, image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "atlas_sources_3d"
ATLAS_VARIANTS = {
    "cortical": "cort-maxprob-thr25-1mm",
    "subcortical": "sub-maxprob-thr25-1mm",
}
DOC_URL = "https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_harvard_oxford.html"
MNI_DOC_URL = "https://nilearn.github.io/stable/modules/generated/nilearn.datasets.load_mni152_template.html"


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_volume(img: nib.spatialimages.SpatialImage, name: str, *, labels: bool = False) -> dict:
    """Check that NIfTI volume has meaningful 3D spatial geometry and voxel data."""
    if len(img.shape) != 3 or any(s < 2 for s in img.shape):
        raise ValueError(f"{name}: expected a 3D volume, got {img.shape}")
    affine = np.asarray(img.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{name}: invalid affine matrix")
    if not np.allclose(affine[3, :], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"{name}: invalid last row of affine")
    if abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError(f"{name}: singular spatial affine")
    voxel_sizes = np.asarray(nib.affines.voxel_sizes(affine))
    if not np.isfinite(voxel_sizes).all() or np.any(voxel_sizes <= 0):
        raise ValueError(f"{name}: invalid voxel dimensions")
    arr = np.asarray(img.dataobj)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: image contains non-finite voxel values")
    unique_ids = None
    if labels:
        if not np.all(arr == np.rint(arr)):
            raise ValueError(f"{name}: atlas contains non-integer label values")
        if np.min(arr) < 0:
            raise ValueError(f"{name}: atlas has negative region identifiers")
        unique_ids = [int(n) for n in np.unique(arr)]
        if len(unique_ids) < 2:
            raise ValueError(f"{name}: atlas contains no meaningful label regions")
    elif float(np.max(arr)) == float(np.min(arr)):
        raise ValueError(f"{name}: template image is constant")

    out = {
        "shape_xyz": [int(x) for x in img.shape],
        "voxel_sizes_mm": [float(x) for x in voxel_sizes],
        "orientation": "".join(nib.aff2axcodes(affine)),
        "affine": affine.tolist(),
        "data_dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    if labels:
        out["present_label_ids"] = unique_ids
    return out


def same_grid(a: nib.spatialimages.SpatialImage, b: nib.spatialimages.SpatialImage) -> bool:
    return a.shape == b.shape and np.allclose(a.affine, b.affine, rtol=0, atol=1e-4)


def label_rows(atlas) -> list[dict]:
    """Get actual numeric atlas indices, preferring Nilearn's explicit LUT."""
    lut = atlas.get("lut") if hasattr(atlas, "get") else getattr(atlas, "lut", None)
    if lut is not None and {"index", "name"}.issubset(lut.columns):
        rows = [{"index": int(idx), "name": str(name)}
                for idx, name in zip(lut["index"], lut["name"])]
    else:
        labels = atlas["labels"] if isinstance(atlas, dict) else atlas.labels
        rows = [{"index": i, "name": str(name)} for i, name in enumerate(labels)]
    if len({r["index"] for r in rows}) != len(rows):
        raise ValueError("Atlas label table contains duplicate numeric indices")
    return sorted(rows, key=lambda r: r["index"])


def save_label_table(path: Path, rows: list[dict], label_data: np.ndarray) -> list[dict]:
    ids, counts = np.unique(label_data, return_counts=True)
    count_map = dict(zip(ids.astype(int).tolist(), counts.astype(int).tolist()))
    defined = {r["index"] for r in rows}
    missing = set(count_map).difference(defined)
    if missing:
        raise ValueError(f"Atlas has voxel IDs missing from Nilearn label table: {sorted(missing)}")
    enriched = [{"index": row["index"], "name": row["name"],
                 "voxels_on_mni_grid": count_map.get(row["index"], 0)} for row in rows]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "name", "voxels_on_mni_grid"])
        writer.writeheader()
        writer.writerows(enriched)
    return enriched


def as_3d_image(obj):
    # Nilearn versions may return either a pathname or a NIfTI image in `maps`.
    return image.load_img(obj)


def save_nifti_checked(img, path: Path, name: str, *, labels=False):
    validate_volume(img, name, labels=labels)
    nib.save(img, str(path))
    loaded = nib.load(str(path))
    validate_volume(loaded, f"saved {name}", labels=labels)
    if not same_grid(img, loaded):
        raise ValueError(f"{name}: saved file lost its spatial grid")
    return loaded


def nearest_label_copy(source_img, template_img):
    # Regridding only. This does NOT perform nonlinear/affine registration.
    resampled = image.resample_to_img(source_img, template_img, interpolation="nearest")
    raw = np.asarray(resampled.dataobj)
    if not np.isfinite(raw).all() or not np.all(raw == np.rint(raw)):
        raise ValueError("Nearest-neighbor resampling unexpectedly created invalid label values")
    unique_before = set(np.unique(np.asarray(source_img.dataobj)).tolist())
    unique_after = set(np.unique(raw).tolist())
    if not unique_after.issubset(unique_before | {0}):
        raise ValueError("Resampling introduced an unknown atlas label")
    hdr = template_img.header.copy()
    hdr.set_data_dtype(np.int16)
    out = nib.Nifti1Image(raw.astype(np.int16), template_img.affine.copy(), header=hdr)
    if not same_grid(out, template_img):
        raise ValueError("Output atlas and MNI template grids do not match")
    return out


def plot_three_planes(template_img, atlas_img, title: str, destination: Path) -> None:
    """QC overlay in RAS+ orientation (axial, coronal, sagittal)."""
    template = np.asarray(nib.as_closest_canonical(template_img).dataobj, dtype=np.float32)
    atlas = np.asarray(nib.as_closest_canonical(atlas_img).dataobj)
    if template.shape != atlas.shape:
        raise ValueError("Canonical template/atlas shapes differ")
    foreground = atlas > 0
    # Pick an informative slice in each plane, rather than an arbitrary midpoint.
    x = int(np.argmax(np.count_nonzero(foreground, axis=(1, 2))))
    y = int(np.argmax(np.count_nonzero(foreground, axis=(0, 2))))
    z = int(np.argmax(np.count_nonzero(foreground, axis=(0, 1))))
    planes = [
        ("Axial", template[:, :, z].T, atlas[:, :, z].T),
        ("Coronal", template[:, y, :].T, atlas[:, y, :].T),
        ("Sagittal", template[x, :, :].T, atlas[x, :, :].T),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    for ax, (name, background, labels) in zip(axes, planes):
        ax.imshow(background, cmap="gray", origin="lower", interpolation="nearest")
        overlay = np.ma.masked_where(labels == 0, labels)
        ax.imshow(overlay, cmap="tab20", alpha=0.5, origin="lower", interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{title} | MNI152 1mm atlas source (QC only)")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination (default: Medsam2_brain/data/atlas_sources_3d)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing previously generated Task 3 outputs")
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    previews = out / "previews"
    cache = out / "nilearn_cache"
    cache.mkdir(exist_ok=True)

    files = {
        "mni_template": out / "mni152_template_1mm.nii.gz",
    }
    for kind in ATLAS_VARIANTS:
        files[f"{kind}_native"] = out / f"harvard_oxford_{kind}_native.nii.gz"
        files[f"{kind}_on_mni"] = out / f"harvard_oxford_{kind}_on_mni152_1mm.nii.gz"
        files[f"{kind}_labels"] = out / f"harvard_oxford_{kind}_labels.csv"
        files[f"{kind}_preview"] = previews / f"harvard_oxford_{kind}_3planes.png"
    files["metadata"] = out / "atlas_source_metadata.json"
    existing = [p for p in files.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Existing outputs in {out} (e.g. {existing[0].name}). "
            "Re-run with --overwrite only if you intend to replace Task 3 outputs."
        )
    previews.mkdir(exist_ok=True)

    print("[1/3] Loading built-in Nilearn MNI152 skull-stripped template at 1 mm...")
    template = datasets.load_mni152_template(resolution=1)
    info = validate_volume(template, "MNI152 template")
    if not np.allclose(info["voxel_sizes_mm"], [1, 1, 1], atol=1e-3, rtol=0):
        raise ValueError(f"MNI template is not 1 mm isotropic: {info['voxel_sizes_mm']}")
    saved_template = save_nifti_checked(template, files["mni_template"], "MNI152 template")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Task 3 - 3D atlas sources (not final 12-region atlas)",
        "project_scope": "Anatomical segmentation only; no stroke analysis",
        "tool_versions": {name: package_version(name) for name in
                          ("numpy", "nibabel", "nilearn", "matplotlib")},
        "mni_template": {
            "variant": "Nilearn load_mni152_template(resolution=1)",
            "source": "Nilearn bundled, skull-stripped / intensity-rescaled MNI152 T1 template",
            "documentation": MNI_DOC_URL,
            "file": files["mni_template"].name,
            "geometry": validate_volume(saved_template, "MNI152 template"),
        },
        "harvard_oxford": {},
        "important_notes": [
            "These are full 3D source atlases, not the final 12-region anatomical label map.",
            "Nearest-neighbor resampling changes voxel grid only; it is not image registration.",
            "Inspect all three planes in both PNGs before using atlas sources in Task 4.",
            "Verify anatomical label IDs and left/right world coordinates in Task 4.",
        ],
    }
    for i, (kind, variant) in enumerate(ATLAS_VARIANTS.items(), start=2):
        print(f"[{i}/3] Fetching Harvard-Oxford {kind}: {variant}...")
        atlas = datasets.fetch_atlas_harvard_oxford(
            variant, data_dir=str(cache), symmetric_split=False
        )
        raw_img = as_3d_image(atlas.maps)
        raw_info = validate_volume(raw_img, f"{kind} atlas (native)", labels=True)
        if not np.allclose(raw_info["voxel_sizes_mm"], [1, 1, 1], atol=1e-3, rtol=0):
            raise ValueError(f"{kind} atlas native source is not 1 mm: {raw_info['voxel_sizes_mm']}")
        native = save_nifti_checked(raw_img, files[f"{kind}_native"], f"{kind} atlas native", labels=True)
        on_mni = nearest_label_copy(native, saved_template)
        saved_on_mni = save_nifti_checked(
            on_mni, files[f"{kind}_on_mni"], f"{kind} atlas on MNI grid", labels=True
        )
        if not same_grid(saved_on_mni, saved_template):
            raise ValueError(f"{kind} atlas does not match MNI template's physical grid")
        rows = save_label_table(
            files[f"{kind}_labels"], label_rows(atlas), np.asarray(saved_on_mni.dataobj)
        )
        plot_three_planes(saved_template, saved_on_mni, f"Harvard-Oxford {kind}", files[f"{kind}_preview"])
        metadata["harvard_oxford"][kind] = {
            "requested_variant": variant,
            "symmetric_split": False,
            "source": "Harvard-Oxford 3D maximum-probability atlas (FSL; distributed via Nilearn)",
            "source_description": str(atlas.get("description", "")) if hasattr(atlas, "get") else "",
            "documentation": DOC_URL,
            "native_file": files[f"{kind}_native"].name,
            "resampled_file": files[f"{kind}_on_mni"].name,
            "label_table_file": files[f"{kind}_labels"].name,
            "label_table": rows,
            "native_geometry": raw_info,
            "mni_grid_geometry": validate_volume(saved_on_mni, f"{kind} on MNI grid", labels=True),
            "resample_interpolation": "nearest",
        }
        print(f"     {len(rows)} labels in LUT, "
              f"{len(metadata['harvard_oxford'][kind]['mni_grid_geometry']['present_label_ids'])} "
              f"present on MNI grid; preview saved.")

    # Metadata excludes the SHA of its own JSON to avoid self-referential hashing.
    metadata["sha256"] = {
        key: sha256_file(path) for key, path in files.items() if key != "metadata"
    }
    files["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n[SUCCESS] Task 3 atlas sources fetched, validated and saved.")
    print(f"Source directory: {out}")
    print(f"Template shape: {saved_template.shape} | voxel size: 1 mm")
    print(f"Previews: {previews}")
    print("Review both 3-plane overlays before beginning Task 4.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
