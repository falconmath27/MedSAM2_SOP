# -*- coding: utf-8 -*-
"""
Run MedSAM2 3D inference + NIfTI conversion for multiple patients/directions.

This script uses environment overrides supported by:
  - inference_medsam2_3d.py
  - 2d_to_nifti.py

Usage:
    python run_batch_inference_and_nifti.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/home/akshaj/Documents/projects/knee_ct_scan_segmentation")
PIPELINE_DIR = PROJECT_ROOT / "pipeline_medsam2_3d"

PATIENTS = [4, 5, 6, 16, 17, 18]
DIRECTIONS = ["axial", "sagittal"]
RUN_SUFFIX = "N10_conf05_iou05"

INFER_SCRIPT = PIPELINE_DIR / "inference_medsam2_3d.py"
NIFTI_SCRIPT = PIPELINE_DIR / "2d_to_nifti.py"


def run_step(cmd: list[str], env: dict[str, str], log_path: Path) -> tuple[int, float]:
    start = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("$ " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode, time.time() - start


def main() -> int:
    logs_dir = PIPELINE_DIR / "batch_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(p, d) for p in PATIENTS for d in DIRECTIONS]
    summary: list[dict[str, object]] = []

    print("=" * 78)
    print("Batch start: inference + nifti conversion")
    print(f"Patients: {PATIENTS}")
    print(f"Directions: {DIRECTIONS}")
    print(f"Run suffix: {RUN_SUFFIX}")
    print(f"Total jobs: {len(jobs)}")
    print("=" * 78)

    base_env = os.environ.copy()

    for i, (patient, direction) in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] Patient p{patient} | {direction}")
        env = base_env.copy()
        env["PATIENT_ID"] = str(patient)
        env["DIRECTION"] = direction
        env["RUN_SUFFIX"] = RUN_SUFFIX

        infer_log = logs_dir / f"p{patient}_{direction}_inference_{RUN_SUFFIX}.log"
        nifti_log = logs_dir / f"p{patient}_{direction}_nifti_{RUN_SUFFIX}.log"

        infer_cmd = [sys.executable, str(INFER_SCRIPT)]
        rc_infer, t_infer = run_step(infer_cmd, env, infer_log)

        item: dict[str, object] = {
            "patient": patient,
            "direction": direction,
            "run_suffix": RUN_SUFFIX,
            "inference_rc": rc_infer,
            "inference_time_sec": round(t_infer, 2),
            "inference_log": str(infer_log),
        }

        if rc_infer != 0:
            print(f"  ❌ Inference failed (rc={rc_infer}). Log: {infer_log}")
            item["status"] = "failed_inference"
            summary.append(item)
            continue

        print(f"  ✅ Inference done in {t_infer:.1f}s")

        nifti_cmd = [sys.executable, str(NIFTI_SCRIPT)]
        rc_nifti, t_nifti = run_step(nifti_cmd, env, nifti_log)

        item.update(
            {
                "nifti_rc": rc_nifti,
                "nifti_time_sec": round(t_nifti, 2),
                "nifti_log": str(nifti_log),
                "output_nifti": str(
                    PIPELINE_DIR
                    / f"p{patient}_3d_{direction}_segmentation_{RUN_SUFFIX}.nii.gz"
                ),
            }
        )

        if rc_nifti != 0:
            print(f"  ❌ NIfTI conversion failed (rc={rc_nifti}). Log: {nifti_log}")
            item["status"] = "failed_nifti"
        else:
            print(f"  ✅ NIfTI done in {t_nifti:.1f}s")
            item["status"] = "ok"

        summary.append(item)

    summary_path = logs_dir / f"batch_summary_{RUN_SUFFIX}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    ok = sum(1 for x in summary if x.get("status") == "ok")
    fail = len(summary) - ok

    print("\n" + "=" * 78)
    print(f"Batch finished. OK: {ok} | Failed: {fail}")
    print(f"Summary: {summary_path}")
    print(f"Logs: {logs_dir}")
    print("=" * 78)

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
