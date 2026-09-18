# -*- coding: utf-8 -*-
"""
dicom_metadata.py — Scan a DICOM directory and generate a CSV mapping
each DICOM filename to its Z position (and other spatial metadata).

This CSV is used by extract_pngs.py, 2d_to_nifti.py, and
postprocess_hu_and_nifti.py to load slices in correct Z order.

Usage:
    python dicom_metadata.py
    python dicom_metadata.py --dicom-dir /path/to/dicoms --output metadata.csv
"""

import argparse
import os
import sys
import time

import pydicom

# =======================================================================
# ========================= DEFAULT CONFIG ==============================
# =======================================================================

PROJECT_ROOT = "/home/akshaj/Documents/projects/knee_ct_scan_segmentation"


# =======================================================================
# ========================= MAIN =======================================
# =======================================================================


def scan_dicom_dir(dicom_dir: str) -> list[dict]:
    """
    Scan a directory for DICOM files and extract spatial metadata.

    Returns a list of dicts sorted by Z position (ascending), each with:
      filename, InstanceNumber, Z, ImagePositionPatient,
      ImageOrientationPatient, SliceThickness, PixelSpacing, Rows, Columns
    """
    entries = []
    all_files = [f for f in os.listdir(dicom_dir) if not f.startswith(".")]

    for fname in all_files:
        fpath = os.path.join(dicom_dir, fname)
        if not os.path.isfile(fpath):
            continue

        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
        except Exception:
            continue  # not a valid DICOM

        # Extract Z position from ImagePositionPatient
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None:
            continue
        z_pos = float(ipp[2])

        iop = getattr(ds, "ImageOrientationPatient", None)
        iop_list = [float(x) for x in iop] if iop is not None else [1, 0, 0, 0, 1, 0]

        ps = getattr(ds, "PixelSpacing", None)
        ps_list = [float(x) for x in ps] if ps is not None else [1.0, 1.0]

        st = getattr(ds, "SliceThickness", None)
        inst = getattr(ds, "InstanceNumber", None)
        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)

        entries.append(
            {
                "filename": fname,
                "InstanceNumber": int(inst) if inst is not None else 0,
                "Z": z_pos,
                "ImagePositionPatient": [float(x) for x in ipp],
                "ImageOrientationPatient": iop_list,
                "SliceThickness": float(st) if st is not None else 1.0,
                "PixelSpacing": ps_list,
                "Rows": int(rows) if rows is not None else 512,
                "Columns": int(cols) if cols is not None else 512,
            }
        )

    # Sort by Z ascending
    entries.sort(key=lambda e: e["Z"])
    return entries


def write_csv(entries: list[dict], output_path: str):
    """Write the metadata list to a CSV file."""
    import csv

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "filename",
        "InstanceNumber",
        "Z",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "SliceThickness",
        "PixelSpacing",
        "Rows",
        "Columns",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            # Stringify list fields for CSV
            row["ImagePositionPatient"] = str(row["ImagePositionPatient"])
            row["ImageOrientationPatient"] = str(row["ImageOrientationPatient"])
            row["PixelSpacing"] = str(row["PixelSpacing"])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Scan DICOM directory and generate metadata CSV with Z positions"
    )
    parser.add_argument(
        "--dicom-dir", type=str, required=True, help="Path to DICOM directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(PROJECT_ROOT, "pipeline_medsam2_3d/dicom_metadata.csv"),
        help="Output CSV path (default: pipeline_medsam2_3d/dicom_metadata.csv)",
    )
    args = parser.parse_args()

    dicom_dir = args.dicom_dir
    output_path = args.output

    print("=" * 70)
    print("  DICOM metadata extraction")
    print("=" * 70)

    if not os.path.isdir(dicom_dir):
        print(f"  [ERROR] DICOM dir not found: {dicom_dir}")
        sys.exit(1)

    print(f"\n[1] Scanning DICOM directory …")
    print(f"    {dicom_dir}")
    t0 = time.time()
    entries = scan_dicom_dir(dicom_dir)
    elapsed = time.time() - t0

    if not entries:
        print("  [ERROR] No valid DICOM files found!")
        sys.exit(1)

    print(f"    Found {len(entries)} DICOM files in {elapsed:.1f}s")
    print(f"    Z range: {entries[0]['Z']} → {entries[-1]['Z']}")
    print(f"    Slice thickness: {entries[0]['SliceThickness']}")
    print(f"    Pixel spacing: {entries[0]['PixelSpacing']}")
    print(f"    Image size: {entries[0]['Rows']}×{entries[0]['Columns']}")

    print(f"\n[2] Writing CSV …")
    write_csv(entries, output_path)
    print(f"    ✅ {output_path}")

    print(f"\n{'=' * 70}")
    print(f"  Done! {len(entries)} slices → {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
