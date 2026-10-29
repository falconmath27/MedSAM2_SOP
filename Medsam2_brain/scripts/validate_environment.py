"""Validate Task 1 and record reproducible environment evidence (no patient data)."""
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def main():
    config = json.loads((ROOT / 'configs/project.json').read_text())
    source = (ROOT / config['medsam2_source']).resolve()
    checkpoint = (ROOT / config['medsam2_checkpoint']).resolve()
    yaml = (ROOT / config['medsam2_config']).resolve()
    for name in ['numpy', 'nibabel', 'nilearn', 'ants', 'SimpleITK', 'PIL',
                 'scipy', 'pandas', 'matplotlib', 'torch', 'torchvision', 'sam2']:
        importlib.import_module(name)
        print(f'Import OK: {name}', flush=True)
    import numpy as np
    import ants
    import psutil
    import torch
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2_video_predictor_npz

    # Exercise the compiled registration library on a small synthetic volume.
    volume = np.zeros((24, 24, 24), dtype=np.float32)
    volume[6:18, 7:17, 8:16] = 1
    fixed = ants.from_numpy(volume)
    registration = ants.registration(fixed=fixed, moving=fixed, type_of_transform='Translation')
    assert np.isfinite(registration['warpedmovout'].numpy()).all()
    assert torch.cuda.is_available(), 'CUDA unavailable'
    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    precision = {}
    for name, dtype in [('float32', torch.float32), ('float16', torch.float16), ('bfloat16', torch.bfloat16)]:
        try:
            x = torch.ones((32, 32), device='cuda', dtype=dtype)
            result = x @ x
            torch.cuda.synchronize()
            precision[name] = bool(torch.isfinite(result).all())
        except RuntimeError as exc:
            precision[name] = str(exc)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(yaml.parent), version_base='1.2'):
        predictor = build_sam2_video_predictor_npz(yaml.name, str(checkpoint), device='cuda',
            hydra_overrides_extra=['++model.fill_hole_area=0'])
    # This builder appends fill_hole_area=8 after caller overrides. Set it explicitly.
    predictor.fill_hole_area = 0
    print('Checkpoint loaded strictly on CUDA', flush=True)
    # Verify both propagation directions from a middle slice.
    frames = torch.zeros((3, 3, predictor.image_size, predictor.image_size), device='cuda')
    seen = set()
    with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
        state = predictor.init_state(frames, 64, 64, offload_state_to_cpu=True)
        for reverse in (False, True):
            predictor.add_new_points_or_box(state, frame_idx=1, obj_id=1,
                                           box=np.array([16, 16, 48, 48], dtype=np.float32))
            for index, ids, logits in predictor.propagate_in_video(state, start_frame_idx=1, reverse=reverse):
                assert logits.shape[-2:] == (64, 64)
                assert torch.isfinite(logits).all()
                seen.add(index)
            predictor.reset_state(state)
    assert seen == {0, 1, 2}, seen
    torch.cuda.synchronize()
    source_files = sorted(p for p in source.rglob('*') if p.is_file() and
                          p.suffix in {'.py', '.yaml', '.yml', '.cu', '.cpp', '.toml'}
                          and not any(part in {'.git', 'build', '__pycache__'} for part in p.relative_to(source).parts))
    source_hashes = {p.relative_to(source).as_posix(): sha256(p) for p in source_files}
    packages = dict(sorted((d.metadata['Name'], d.version) for d in metadata.distributions()))
    report = {
        'validated_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'packages': packages, 'gpu': torch.cuda.get_device_name(0),
        'compute_capability': torch.cuda.get_device_capability(0), 'torch_cuda': torch.version.cuda,
        'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
        'gpu_free_bytes_before_model': free, 'gpu_total_bytes': total,
        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'ram_available_bytes': psutil.virtual_memory().available,
        'ram_total_bytes': psutil.virtual_memory().total, 'precision_matmul_checks': precision,
        'bf16_supported': torch.cuda.is_bf16_supported(), 'smoke_precision': str(dtype),
        'fill_hole_area': predictor.fill_hole_area,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'yaml': str(yaml), 'yaml_sha256': sha256(yaml), 'source_directory': str(source),
        'containing_git_root': git(source, 'rev-parse', '--show-toplevel'),
        'containing_git_revision': git(source, 'rev-parse', 'HEAD'),
        'source_git_status': git(source, 'status', '--short', '--', '.'),
        'source_file_sha256': source_hashes,
        'checks': ['imports', 'ANTs translation registration', 'strict checkpoint load',
                   'three-slice bidirectional CUDA propagation'],
        'limitations': ['Synthetic smoke test, not anatomical accuracy validation',
                       'Optional SAM2 CUDA connected-components extension disabled; fill_hole_area=0',
                       'Windows attention kernels can fall back to standard PyTorch attention',
                       'Git revision identifies the containing SOP repository, not an independent upstream MedSAM2 checkout']
    }
    destination = ROOT / 'environment'
    destination.mkdir(exist_ok=True)
    (destination / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (destination / 'requirements-lock.txt').write_text(frozen)
    print(json.dumps({k: report[k] for k in ['status', 'gpu', 'gpu_peak_allocated_bytes', 'smoke_precision', 'checkpoint_sha256']}, indent=2))


if __name__ == '__main__':
    main()
