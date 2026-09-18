# -*- coding: utf-8 -*-
"""
compute_nifti_iou.py

Compute IoU between two binary 3D mask NIfTI files:
- Overall IoU
- Left/right-side IoU
- Per-group IoU across slice ranges

Notes on direction/orientation:
- If masks come from different directions (axial/coronal/sagittal outputs),
  orientation and voxel grids may differ even if shapes match.
- Use --align (default) to canonicalize and resample mask B onto mask A grid
  before IoU calculation.

Usage examples:
python compute_nifti_iou.py \
  --mask-a a.nii.gz --mask-b b.nii.gz

python compute_nifti_iou.py \
  --mask-a a.nii.gz --mask-b b.nii.gz \
  --slice-axis axial --num-groups 6 --left-right-axis 0 \
  --output-json iou_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


AXIS_NAME_TO_INDEX = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}


def parse_axis(value: str) -> int:
    v = value.strip().lower()
    if v in AXIS_NAME_TO_INDEX:
        return AXIS_NAME_TO_INDEX[v]
    if v in {"0", "1", "2"}:
        return int(v)
    raise ValueError(
        f"Invalid axis '{value}'. Use one of: axial, coronal, sagittal, 0, 1, 2"
    )


def format_metric(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x:.6f}"


def ensure_3d(img: nib.Nifti1Image, label: str) -> None:
    if len(img.shape) < 3:
        raise ValueError(f"{label} is not 3D (shape={img.shape})")
    if len(img.shape) > 3:
        raise ValueError(
            f"{label} has >3 dims (shape={img.shape}). Please pass a 3D mask NIfTI."
        )


def load_binary_mask(path: Path) -> nib.Nifti1Image:
    img = nib.load(str(path))
    ensure_3d(img, str(path))
    return img


def maybe_align_images(
    img_a: nib.Nifti1Image,
    img_b: nib.Nifti1Image,
    align: bool,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    if not align:
        return img_a, img_b

    # Canonical orientation (RAS-like) first, then grid match by nearest-neighbor resampling.
    a_can = nib.as_closest_canonical(img_a)
    b_can = nib.as_closest_canonical(img_b)

    same_shape = a_can.shape == b_can.shape
    same_affine = np.allclose(a_can.affine, b_can.affine)

    if same_shape and same_affine:
        return a_can, b_can

    b_res = resample_from_to(
        b_can, (a_can.shape, a_can.affine), order=0, mode="nearest"
    )
    return a_can, b_res


def to_bool_array(img: nib.Nifti1Image) -> np.ndarray:
    arr = np.asanyarray(img.dataobj)
    return arr > 0


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, int, int]:
    inter = int(np.count_nonzero(mask_a & mask_b))
    union = int(np.count_nonzero(mask_a | mask_b))
    if union == 0:
        return float("nan"), inter, union
    return inter / union, inter, union


def side_view(mask: np.ndarray, axis: int, side: str) -> np.ndarray:
    mid = mask.shape[axis] // 2
    sl = [slice(None)] * 3
    if side == "left":
        sl[axis] = slice(0, mid)
    elif side == "right":
        sl[axis] = slice(mid, None)
    else:
        raise ValueError(f"Invalid side '{side}'")
    return mask[tuple(sl)]


def group_ranges(length: int, num_groups: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, length, num_groups + 1).astype(int)
    out = []
    for i in range(len(edges) - 1):
        s, e = int(edges[i]), int(edges[i + 1])
        if e > s:
            out.append((s, e))
    return out


def slice_group(mask: np.ndarray, axis: int, start: int, end: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[axis] = slice(start, end)
    return mask[tuple(sl)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute IoU between two NIfTI masks")
    p.add_argument("--mask-a", required=True, help="Path to mask A (reference grid)")
    p.add_argument("--mask-b", required=True, help="Path to mask B")
    p.add_argument(
        "--slice-axis",
        default="axial",
        help="Axis for grouped-slice IoU: axial/coronal/sagittal or 0/1/2 (default: axial)",
    )
    p.add_argument(
        "--num-groups",
        type=int,
        default=6,
        help="Number of slice groups for per-group IoU (default: 6)",
    )
    p.add_argument(
        "--left-right-axis",
        default="0",
        help="Axis used to split left/right halves (default: 0; typically L-R in canonical)",
    )
    p.add_argument(
        "--no-align",
        action="store_true",
        help="Disable canonicalization/resampling alignment before IoU",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save full results as JSON",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mask_a_path = Path(args.mask_a)
    mask_b_path = Path(args.mask_b)

    if not mask_a_path.is_file():
        raise FileNotFoundError(f"Mask A not found: {mask_a_path}")
    if not mask_b_path.is_file():
        raise FileNotFoundError(f"Mask B not found: {mask_b_path}")

    num_groups = int(args.num_groups)
    if num_groups <= 0:
        raise ValueError("--num-groups must be > 0")

    group_axis = parse_axis(str(args.slice_axis))
    lr_axis = parse_axis(str(args.left_right_axis))
    do_align = not args.no_align

    img_a = load_binary_mask(mask_a_path)
    img_b = load_binary_mask(mask_b_path)
    img_a, img_b = maybe_align_images(img_a, img_b, align=do_align)

    mask_a = to_bool_array(img_a)
    mask_b = to_bool_array(img_b)

    if mask_a.shape != mask_b.shape:
        raise RuntimeError(
            f"Shape mismatch after alignment step: {mask_a.shape} vs {mask_b.shape}"
        )

    if not np.allclose(img_a.affine, img_b.affine):
        print("[WARN] Affines still differ; IoU may be unreliable.")

    overall_iou, overall_inter, overall_union = iou(mask_a, mask_b)

    left_a = side_view(mask_a, axis=lr_axis, side="left")
    left_b = side_view(mask_b, axis=lr_axis, side="left")
    right_a = side_view(mask_a, axis=lr_axis, side="right")
    right_b = side_view(mask_b, axis=lr_axis, side="right")

    left_iou, left_inter, left_union = iou(left_a, left_b)
    right_iou, right_inter, right_union = iou(right_a, right_b)

    ranges = group_ranges(mask_a.shape[group_axis], num_groups=num_groups)
    group_rows = []
    for idx, (start, end) in enumerate(ranges, start=1):
        a_g = slice_group(mask_a, axis=group_axis, start=start, end=end)
        b_g = slice_group(mask_b, axis=group_axis, start=start, end=end)

        iou_all, inter_all, union_all = iou(a_g, b_g)

        iou_l, inter_l, union_l = iou(
            side_view(a_g, axis=lr_axis, side="left"),
            side_view(b_g, axis=lr_axis, side="left"),
        )
        iou_r, inter_r, union_r = iou(
            side_view(a_g, axis=lr_axis, side="right"),
            side_view(b_g, axis=lr_axis, side="right"),
        )

        group_rows.append(
            {
                "group": idx,
                "slice_start": start,
                "slice_end_exclusive": end,
                "iou_overall": iou_all,
                "iou_left": iou_l,
                "iou_right": iou_r,
                "intersection_overall": inter_all,
                "union_overall": union_all,
                "intersection_left": inter_l,
                "union_left": union_l,
                "intersection_right": inter_r,
                "union_right": union_r,
            }
        )

    result = {
        "mask_a": str(mask_a_path),
        "mask_b": str(mask_b_path),
        "aligned": do_align,
        "shape_used": list(mask_a.shape),
        "group_axis": group_axis,
        "left_right_axis": lr_axis,
        "num_groups_requested": num_groups,
        "num_groups_used": len(group_rows),
        "overall": {
            "iou": overall_iou,
            "intersection": overall_inter,
            "union": overall_union,
        },
        "left": {
            "iou": left_iou,
            "intersection": left_inter,
            "union": left_union,
        },
        "right": {
            "iou": right_iou,
            "intersection": right_inter,
            "union": right_union,
        },
        "groups": group_rows,
    }

    print("=" * 80)
    print("NIfTI IoU Report")
    print("=" * 80)
    print("Mask A:", mask_a_path)
    print("Mask B:", mask_b_path)
    print("Aligned:", do_align)
    print("Shape used:", mask_a.shape)
    print("Group axis:", group_axis, "| Left-right axis:", lr_axis)
    print("-")
    print(
        f"Overall IoU: {format_metric(overall_iou)} "
        f"(inter={overall_inter}, union={overall_union})"
    )
    print(
        f"Left IoU:    {format_metric(left_iou)} "
        f"(inter={left_inter}, union={left_union})"
    )
    print(
        f"Right IoU:   {format_metric(right_iou)} "
        f"(inter={right_inter}, union={right_union})"
    )

    print("-")
    print("Per-group IoU:")
    for row in group_rows:
        s = row["slice_start"]
        e = row["slice_end_exclusive"]
        print(
            f"  Group {row['group']:>2} [{s:>4}:{e:<4}] | "
            f"all={format_metric(row['iou_overall'])} "
            f"left={format_metric(row['iou_left'])} "
            f"right={format_metric(row['iou_right'])}"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("-")
        print("Saved JSON:", out_path)


if __name__ == "__main__":
    main()
