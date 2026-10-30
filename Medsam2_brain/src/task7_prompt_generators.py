"""Task 7 — CT-only atlas registration and automatic 3D anatomical prompts.

From Medsam2_brain (with the validated .venv activated):

    python src/step4_prompt_generators.py --list-development
    python src/step4_prompt_generators.py --patient train_040
    python src/step4_prompt_generators.py --all-development

Inputs:  Task 2 manifest + Task 4 MNI152 template and anatomy_12regions_atlas.
Outputs: data/prompts_json/modeA_atlas/<patient_id>/
    warped_atlas_on_ct.nii.gz        integer labels 0..12, EXACT original CT grid
    mni_registered_to_ct.nii.gz      registration inspection only
    transforms/*                    forward and inverse ANTs transforms
    prompts.json                    Task 8 box prompts in ORIGINAL CT pixel coordinates
    qc_report.json                  region counts, QC flags, settings
    previews/ct_atlas_3planes.png
    previews/region_key_slices.png
    previews/ct_mni_registration_3planes.png

Coordinates and contract (same as Task 6):
    Original NIfTI:   (i, j, k) == (x_index, y_index, slice_index)
    CT/MedSAM2:      (z, y, x) == (k, j, i), NO image flips/reorientations
    Prompt slice:    native k index, 0 <= k < K (can be inferior-to-superior
                     OR superior-to-inferior depending on NIfTI affine)
    Prompt box:     [x1, y1, x2, y2] in ORIGINAL CT pixel coordinates,
                    inclusive extrema, not resized 512x512 coordinates.
    z_min/z_max:     inclusive slice bounds, z_key is largest-region-area slice.
    Only CT, MNI and atlas are read; NO ADC, MRI references or lesion masks.

The default linear 'Affine' transform is an INITIAL baseline, NOT deformable.
An optional 'SyN' run provides nonlinear registration when indicated by QC.
Registration can be wrong despite completion and valid metadata: QC is mandatory.
Requires numpy, nibabel, matplotlib, antspyx (import ants).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'data' / 'manifests' / 'patients.json'
DEFAULT_TEMPLATE = ROOT / 'data' / 'master_atlas_3d' / 'mni_template_1mm.nii.gz'
DEFAULT_ATLAS = ROOT / 'data' / 'master_atlas_3d' / 'anatomy_12regions_atlas.nii.gz'
DEFAULT_OUTPUT = ROOT / 'data' / 'prompts_json' / 'modeA_atlas'
REGIONS = {
    1: 'Left precentral gyrus', 2: 'Right precentral gyrus',
    3: 'Left insular cortex', 4: 'Right insular cortex',
    5: 'Left caudate', 6: 'Right caudate',
    7: 'Left lentiform (putamen + pallidum)',
    8: 'Right lentiform (putamen + pallidum)',
    9: 'Left thalamus', 10: 'Right thalamus',
    11: 'Left lateral ventricle', 12: 'Right lateral ventricle',
}
COLORS = [
    '#d65057', '#f29a9d', '#df9237', '#f0d491', '#44a186', '#90d2b8',
    '#527dd2', '#a2b5e6', '#8c68c6', '#bba6e1', '#3185ad', '#82c6de',
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_id(s: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', s):
        raise ValueError(f'Invalid patient id {s!r}')
    return s


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Task 2 manifest missing: {path}')
    document = json.loads(path.read_text(encoding='utf-8'))
    patients = document.get('patients')
    if not isinstance(patients, list):
        raise ValueError('Manifest must contain a patients list')
    keys = [p.get('patient_id') for p in patients]
    if None in keys or len(keys) != len(set(keys)):
        raise ValueError('Patient IDs missing or duplicated')
    return document


def select_patients(document: dict, patient_id: str | None,
                    all_development: bool, allow_reserved: bool) -> list[dict]:
    patients = document['patients']
    if patient_id:
        chosen = [p for p in patients if p['patient_id'] == patient_id]
        if not chosen:
            raise ValueError(f'Patient {patient_id!r} not found in manifest')
    else:
        dev = sorted((p for p in patients if p.get('split') == 'development'),
                     key=lambda p: p['patient_id'])
        if not dev:
            raise ValueError('No development patients in Task 2 manifest')
        chosen = dev if all_development else dev[:1]
    for p in chosen:
        safe_id(p['patient_id'])
        if p.get('split') != 'development' and not allow_reserved:
            raise ValueError(f"{p['patient_id']} is {p.get('split')!r}; run development first. "
                             'Use --allow-reserved only after freezing settings.')
        if p.get('issues') or p.get('status') == 'needs_attention':
            raise ValueError(f"{p['patient_id']} has unresolved Task 2 issues: {p.get('issues')}")
        ct = p.get('ncct')
        if not isinstance(ct, dict) or not ct.get('path') or ct.get('valid') is False:
            raise ValueError(f"{p['patient_id']}: missing validated NCCT input")
        if not Path(ct['path']).is_file():
            raise FileNotFoundError(f"{p['patient_id']} CT not found: {ct['path']}")
    return chosen


def validate_master(template_path: Path, atlas_path: Path):
    import nibabel as nib
    if not template_path.is_file() or not atlas_path.is_file():
        raise FileNotFoundError('Task 4 MNI template/12-label atlas missing; run Task 4 first')
    template = nib.load(str(template_path))
    atlas = nib.load(str(atlas_path))
    if len(template.shape) != 3 or template.shape != atlas.shape:
        raise ValueError('Atlas and MNI must both be 3D and on same voxel grid')
    if not np.allclose(template.affine, atlas.affine, atol=1e-4, rtol=0):
        raise ValueError('Master atlas and MNI affines differ')
    v = np.asarray(atlas.dataobj)
    if not np.isfinite(v).all() or not np.allclose(v, np.rint(v), atol=1e-5):
        raise ValueError('Master atlas has non-integer / non-finite labels')
    labels = set(np.unique(v).astype(int).tolist())
    if labels != set(range(13)):
        raise ValueError(f'Task 4 master atlas must have labels 0..12; found {sorted(labels)}')
    return template, atlas


def load_ct(path: Path):
    import nibabel as nib
    ct = nib.load(str(path))
    if len(ct.shape) != 3 or min(ct.shape) < 3:
        raise ValueError(f'CT must be nonempty 3D, got {ct.shape}')
    if not np.isfinite(ct.affine).all() or abs(np.linalg.det(ct.affine[:3,:3])) < 1e-8:
        raise ValueError('CT spatial affine is invalid')
    hu = np.asarray(ct.dataobj, dtype=np.float32)  # honors NIfTI slope/intercept
    if not np.isfinite(hu).all():
        raise ValueError('CT has NaN/Inf voxels')
    if (np.sum((hu >= 0) & (hu <= 100)) < 10):
        raise ValueError('CT has no plausible 0..100 HU values; verify original HU scaling')
    return ct, hu


def validate_task6_metadata(patient_id: str, ct, ct_path: Path, task6_root: Path) -> str:
    """Optional integration check. Does not read/use Task 6 windows for registration."""
    info = task6_root / patient_id / 'metadata.json'
    if not info.exists():
        return 'not_present; Task 7 reads original CT independently'
    metadata = json.loads(info.read_text(encoding='utf-8'))
    if tuple(metadata.get('original_shape_xyz', ())) != tuple(ct.shape):
        raise ValueError(f'{patient_id}: Task 6 CT shape disagrees with current CT')
    saved_affine = np.asarray(metadata.get('original_affine_voxel_xyz_to_world_ras_mm'))
    if saved_affine.shape != (4, 4) or not np.allclose(saved_affine, ct.affine, atol=1e-4, rtol=0):
        raise ValueError(f'{patient_id}: Task 6 affine disagrees with current CT')
    existing = metadata.get('source_ct_path')
    if existing and Path(existing).resolve() != ct_path.resolve():
        raise ValueError(f'{patient_id}: Task 6 and Task 7 are using different CT source paths')
    return 'CT path, native shape and affine match Task 6'


def preprocess_ct_for_registration(raw: np.ndarray) -> np.ndarray:
    """Fixed CT moving-to-MNI multimodal metric image (brain soft tissue only).

    Out-of-window air / very dense bone become background. Between 0 and 80 HU,
    values are retained as a gradient for Mattes mutual information registration.
    Original CT is never modified or saved in transformed HU units.
    """
    result = np.clip(raw, 0.0, 80.0) / 80.0
    result = np.where((raw >= -5.0) & (raw <= 110.0), result, 0.0)
    return np.asarray(result, dtype=np.float32)


def preprocess_t1_for_registration(raw: np.ndarray) -> np.ndarray:
    """Scale template nonzero brain intensities for multimodal MI; preserve geometry."""
    good = raw[np.isfinite(raw) & (raw > 0)]
    if good.size < 100:
        raise ValueError('MNI template has too few positive voxels')
    lo, hi = np.percentile(good, [1, 99])
    if hi <= lo:
        raise ValueError('MNI template is constant')
    out = np.clip((raw - float(lo)) / float(hi-lo), 0.0, 1.0)
    out[raw <= 0] = 0
    return np.asarray(out, dtype=np.float32)


def extract_prompts(mask_zyx: np.ndarray, pad: int = 3) -> tuple[dict, list[int], dict]:
    """Build per-region prompts; boxes are INCLUSIVE original CT pixel extrema."""
    if mask_zyx.ndim != 3 or min(mask_zyx.shape) <= 1:
        raise ValueError(f'Expected label image (Z,H,W), got {mask_zyx.shape}')
    if pad < 0:
        raise ValueError('pad must be nonnegative')
    Z, H, W = mask_zyx.shape
    prompts: dict[str, dict] = {}
    missing = []
    counts = {}
    for rid in range(1, 13):
        binary = mask_zyx == rid
        n = int(np.count_nonzero(binary))
        counts[str(rid)] = n
        if n == 0:
            missing.append(rid)
            continue
        slice_areas = np.count_nonzero(binary, axis=(1, 2))
        seen = np.flatnonzero(slice_areas)
        zmin, zmax = int(seen[0]), int(seen[-1])
        zkey = int(np.argmax(slice_areas))
        ys, xs = np.where(binary[zkey])
        x1 = max(0, int(xs.min()) - pad)
        x2 = min(W - 1, int(xs.max()) + pad)
        y1 = max(0, int(ys.min()) - pad)
        y2 = min(H - 1, int(ys.max()) + pad)
        assert 0 <= zmin <= zkey <= zmax < Z
        assert 0 <= x1 <= x2 < W and 0 <= y1 <= y2 < H
        prompts[str(rid)] = {
            'region_id': rid, 'region_name': REGIONS[rid],
            'z_min': zmin, 'z_max': zmax, 'z_key': zkey,
            'box': [x1, y1, x2, y2],
            'z_key_area_pixels': int(slice_areas[zkey]),
            'total_voxels': n,
            'boxes_are': 'inclusive original CT pixel-center indices [x1,y1,x2,y2]',
        }
    return prompts, missing, counts


def spatial_qc(labels_xyz: np.ndarray, ct_affine: np.ndarray,
               prompts: dict, counts: dict, min_voxels: int) -> dict:
    """Conservative *heuristic* flags; never claim anatomical accuracy."""
    import nibabel as nib
    flags = []
    axes = labels_xyz.shape
    n = int(np.prod(axes))
    pct = float(np.count_nonzero(labels_xyz) / n)
    if pct < 0.005 or pct > 0.55:
        flags.append(f'Anatomical label fraction {pct:.3%} outside broad 0.5–55% QC bounds')
    for i in range(1, 13):
        c = counts[str(i)]
        if c == 0:
            flags.append(f'{i}: {REGIONS[i]} is absent on CT grid')
        elif c < min_voxels:
            flags.append(f'{i}: {REGIONS[i]} has only {c} voxels (<{min_voxels})')
        if str(i) in prompts:
            x1, y1, x2, y2 = prompts[str(i)]['box']
            if x1 == 0 or y1 == 0 or x2 == axes[0]-1 or y2 == axes[1]-1:
                flags.append(f'{i}: key-slice bounding box touches CT image edge')
            pp = prompts[str(i)]
            if pp['z_min'] == 0 or pp['z_max'] == axes[2]-1:
                flags.append(f'{i}: region touches first or last CT slice')
    centroids = {}
    for rid in range(1, 13):
        xyz = np.argwhere(labels_xyz == rid)
        if len(xyz):
            ctr = xyz.mean(axis=0)
            centroids[str(rid)] = [round(float(v), 3)
                                   for v in nib.affines.apply_affine(ct_affine, ctr)]
    for rid in (1, 3, 5, 7, 9, 11):
        a, b = centroids.get(str(rid)), centroids.get(str(rid+1))
        # RAS world X increases toward patient's right; actual CT origin need not be 0.
        if a and b and a[0] >= b[0]:
            flags.append(f'Left/right region centroids reversed for IDs {rid}/{rid+1} in RAS world X')
    return {'status': 'review_required', 'flags': flags,
            'label_fraction_of_ct': round(pct, 6),
            'region_centroids_world_ras_mm': centroids,
            'note': 'These QC checks do NOT measure registration accuracy. Verify anatomical overlays. '
                    'Atlas-registered prompts may share source bias with MRI-derived references.'}


def _plot_base(ax, ct_slice, labels_slice, extent, title, cmap=None):
    ax.imshow(ct_slice, origin='lower', cmap='gray', vmin=0, vmax=80,
              extent=extent, interpolation='nearest', aspect='equal')
    if labels_slice is not None:
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cs = ['#00000000', *COLORS]
        custom = ListedColormap(cs)
        overlay = np.ma.masked_where(labels_slice == 0, labels_slice)
        ax.imshow(overlay, origin='lower', cmap=custom,
                  norm=BoundaryNorm(np.arange(-0.5, 13.5, 1), custom.N),
                  extent=extent, interpolation='nearest', alpha=0.6, aspect='equal')
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def save_three_planes(ct_xyz: np.ndarray, label_xyz: np.ndarray,
                      ct_affine: np.ndarray, destination: Path, patient: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nibabel as nib
    nx, ny, nz = ct_xyz.shape
    sx, sy, sz = nib.affines.voxel_sizes(ct_affine)
    extents = [(0, nx*sx, 0, ny*sy), (0, nx*sx, 0, nz*sz), (0, ny*sy, 0, nz*sz)]
    i, j, k = nx//2, ny//2, int(np.argmax(np.count_nonzero(label_xyz != 0, axis=(0,1))))
    ct_slices = [ct_xyz[:,:,k].T, ct_xyz[:,j,:].T, ct_xyz[i,:,:].T]
    labels = [label_xyz[:,:,k].T, label_xyz[:,j,:].T, label_xyz[i,:,:].T]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, c, l, ex, title in zip(axes, ct_slices, labels, extents,
                                  [f'Axial k={k}', f'Coronal j={j}', f'Sagittal i={i}']):
        _plot_base(ax, np.clip(c,0,80), l, ex, title)
    fig.suptitle(f'{patient} | CT + warped 12-region atlas | native voxel axes; mm aspect ratio')
    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_key_slices(ct_xyz: np.ndarray, labels_xyz: np.ndarray,
                   prompts: dict, destination: Path, patient: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    for rid, ax in enumerate(axes.flat, start=1):
        region = prompts.get(str(rid))
        if region is None:
            ax.set_title(f'{rid}: MISSING {REGIONS[rid]}', fontsize=9)
            ax.axis('off')
            continue
        k = region['z_key']
        raw = ct_xyz[:,:,k].T
        label = labels_xyz[:,:,k].T == rid
        ax.imshow(np.clip(raw, 0, 80), origin='lower', cmap='gray', vmin=0, vmax=80)
        layer = np.ma.masked_where(~label, label.astype(float))
        ax.imshow(layer, cmap='Reds', origin='lower', vmin=0, vmax=1, alpha=.6)
        x1,y1,x2,y2 = region['box']
        ax.add_patch(Rectangle((x1-.5, y1-.5), x2-x1+1, y2-y1+1,
                               fill=False, edgecolor='#34d5f0', lw=1.1))
        ax.set_title(f'{rid}: {REGIONS[rid]}  | k={k}', fontsize=9)
        ax.axis('off')
    fig.suptitle(f'{patient} | Region key slices and cyan automated bounding boxes', fontsize=13)
    fig.tight_layout()
    fig.savefig(destination, dpi=145, bbox_inches='tight')
    plt.close(fig)


def save_registration_preview(ct_xyz: np.ndarray, warped_t1_xyz: np.ndarray,
                              affine: np.ndarray, destination: Path, patient: str):
    """Axial/coronal/sagittal raw CT vs registered MNI for manual visual QC."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nibabel as nib
    nx,ny,nz = ct_xyz.shape
    sx,sy,sz = nib.affines.voxel_sizes(affine)
    extents = [(0,nx*sx,0,ny*sy), (0,nx*sx,0,nz*sz), (0,ny*sy,0,nz*sz)]
    i,j,k = nx//2, ny//2, nz//2
    c = [ct_xyz[:,:,k].T, ct_xyz[:,j,:].T, ct_xyz[i,:,:].T]
    t = [warped_t1_xyz[:,:,k].T, warped_t1_xyz[:,j,:].T, warped_t1_xyz[i,:,:].T]
    positive = warped_t1_xyz[np.isfinite(warped_t1_xyz) & (warped_t1_xyz>0)]
    lo, hi = np.percentile(positive, [2, 98]) if positive.size else (0., 1.)
    if hi <= lo: hi = lo+1
    fig, ax = plt.subplots(2, 3, figsize=(14, 9))
    for n, name in enumerate(['Axial','Coronal','Sagittal']):
        _plot_base(ax[0,n], np.clip(c[n],0,80), None, extents[n], f'{name}: CT')
        ax[1,n].imshow(t[n], cmap='gray', origin='lower', vmin=lo, vmax=hi,
                       extent=extents[n], interpolation='nearest', aspect='equal')
        ax[1,n].set_title(f'{name}: registered MNI T1', fontsize=10)
        ax[1,n].axis('off')
    fig.suptitle(f'{patient} | fixed CT vs registered MNI: CHECK anatomy, not intensity match')
    fig.tight_layout()
    fig.savefig(destination, dpi=145, bbox_inches='tight')
    plt.close(fig)


def ensure_on_original_grid(path: Path, ct, strict_tol: float = 0.015):
    """Validate ANTs-saved output geometry before writing with original CT header.

    NEVER attach the CT affine to a volume whose spatial grid is different.
    """
    import nibabel as nib
    img = nib.load(str(path))
    if img.shape != ct.shape or not np.allclose(img.affine, ct.affine, atol=strict_tol, rtol=0):
        delta = float(np.max(np.abs(img.affine - ct.affine))) if img.shape == ct.shape else None
        raise ValueError(f'Warped output does NOT match CT geometry; shapes: {img.shape} / {ct.shape}, '
                         f'affine maximum delta: {delta}. Do NOT relabel with CT affine blindly.')
    return img


def run_patient(patient: dict, mni_path: Path, atlas_path: Path, output_root: Path,
                args) -> tuple[Path, dict]:
    import ants
    import nibabel as nib

    pid = safe_id(patient['patient_id'])
    ct_path = Path(patient['ncct']['path']).resolve()
    final = output_root / pid
    if final.exists() and any(final.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{final} already contains output files; use --overwrite to rebuild')
    ct_nib, hu = load_ct(ct_path)
    source_agreement = validate_task6_metadata(pid, ct_nib, ct_path, args.task6_root)
    stage = Path(tempfile.mkdtemp(prefix=f'.partial_{pid}_', dir=output_root))
    try:
        (stage/'transforms').mkdir()
        (stage/'previews').mkdir()
        print(f'[{pid}] CT-only MNI -> CT ({args.registration}, Mattes mutual information) ...', flush=True)
        ct_ants = ants.image_read(str(ct_path))
        t1_ants = ants.image_read(str(mni_path))
        atlas_ants = ants.image_read(str(atlas_path))
        fixed = ct_ants.new_image_like(preprocess_ct_for_registration(ct_ants.numpy()))
        moving = t1_ants.new_image_like(preprocess_t1_for_registration(t1_ants.numpy()))
        if float(np.std(fixed.numpy())) < 1e-5:
            raise ValueError('CT preprocessing produced nearly constant image')
        transform_prefix = str(stage/'transforms'/'mni_to_ct_')
        reg = ants.registration(
            fixed=fixed, moving=moving, type_of_transform=args.registration,
            outprefix=transform_prefix, aff_metric='mattes', aff_sampling=32,
            aff_random_sampling_rate=0.3, syn_metric='mattes', syn_sampling=32,
            verbose=args.verbose,
        )
        if not reg.get('fwdtransforms'):
            raise RuntimeError('ANTs registration returned no forward transforms')
        print(f'[{pid}] Warping atlas labels to ORIGINAL CT grid ...', flush=True)
        label_image = ants.apply_transforms(
            fixed=ct_ants, moving=atlas_ants, transformlist=reg['fwdtransforms'],
            interpolator='genericLabel',
        )
        warped_tmp = stage/'_warped_ants_tmp.nii.gz'
        ants.image_write(label_image, str(warped_tmp))
        warped_nib = ensure_on_original_grid(warped_tmp, ct_nib)
        warped_values = np.asarray(warped_nib.dataobj, dtype=np.float32)
        if not np.isfinite(warped_values).all() or not np.allclose(warped_values, np.rint(warped_values), atol=1e-3):
            raise ValueError('Warped atlas contains invalid/interpolated non-integer labels')
        labels_xyz = np.rint(warped_values).astype(np.int16)
        if labels_xyz.min() < 0 or labels_xyz.max() > 12:
            raise ValueError('Warped atlas contains label values outside 0..12')
        labels_xyz = labels_xyz.astype(np.uint8)
        output_atlas = stage/'warped_atlas_on_ct.nii.gz'
        header = ct_nib.header.copy()
        header.set_data_dtype(np.uint8)
        header.set_slope_inter(1.0, 0.0)
        nib.save(nib.Nifti1Image(labels_xyz, ct_nib.affine, header=header), str(output_atlas))
        verify = nib.load(str(output_atlas))
        if verify.shape != ct_nib.shape or not np.allclose(verify.affine, ct_nib.affine, atol=1e-4, rtol=0):
            raise RuntimeError('Final NIfTI geometry changed unexpectedly')
        if not np.array_equal(np.asarray(verify.dataobj), labels_xyz):
            raise RuntimeError('Saved label data differs from in-memory map')
        warped_tmp.unlink()

        reg_image = reg['warpedmovout']
        mni_out = stage/'mni_registered_to_ct.nii.gz'
        ants.image_write(reg_image, str(mni_out))
        warped_t1_nib = ensure_on_original_grid(mni_out, ct_nib)
        warped_t1 = np.asarray(warped_t1_nib.dataobj, dtype=np.float32)
        mask_zyx = np.transpose(labels_xyz, (2,1,0))
        prompts, missing, counts = extract_prompts(mask_zyx, args.pad)
        qc = spatial_qc(labels_xyz, ct_nib.affine, prompts, counts, args.min_voxels)
        if len(prompts) == 0:
            raise RuntimeError('Registration produced no target anatomy anywhere in CT FOV')
        print(f'[{pid}] Extracted {len(prompts)}/12 anatomical prompts; creating previews ...', flush=True)
        save_three_planes(hu, labels_xyz, ct_nib.affine, stage/'previews'/'ct_atlas_3planes.png', pid)
        save_key_slices(hu, labels_xyz, prompts, stage/'previews'/'region_key_slices.png', pid)
        save_registration_preview(hu, warped_t1, ct_nib.affine,
                                  stage/'previews'/'ct_mni_registration_3planes.png', pid)
        transform_files = sorted(p for p in (stage/'transforms').rglob('*') if p.is_file())
        report = {
            'task': 7, 'created_utc': now(), 'patient_id': pid, 'mode': 'A_atlas_CT_only',
            'input_modalities': ['NCCT', 'MNI152 T1 template', 'Harvard-Oxford anatomical atlas'],
            'MRI_ADC_lesion_inputs_used': False,
            'source_ct': str(ct_path), 'ct_sha256': sha256(ct_path),
            'mni_template': str(mni_path), 'mni_sha256': sha256(mni_path),
            'atlas': str(atlas_path), 'atlas_sha256': sha256(atlas_path),
            'task6_consistency': source_agreement,
            'registration': {
                'method': args.registration,
                'fixed': 'native CT', 'moving': 'MNI152 T1 template',
                'metric': 'Mattes mutual information (affine; also nonlinear if SyN)',
                'ct_preprocessing': '0..80 HU, suppress HU <-5 or >110; raw CT unchanged',
                'mni_preprocessing': 'nonzero T1 percentile 1..99 -> [0,1]',
                'registration_output_transforms': [str(p.relative_to(stage)) for p in transform_files],
                'forward_transform_order': [str(Path(p).relative_to(stage)) for p in reg['fwdtransforms']],
                'inverse_transform_order': [str(Path(p).relative_to(stage)) for p in reg.get('invtransforms', [])],
                'atlas_interpolation': 'genericLabel',
            },
            'original_CT_shape_xyz': list(ct_nib.shape),
            'original_CT_shape_zyx': list(map(int, mask_zyx.shape)),
            'original_CT_affine': ct_nib.affine.tolist(),
            'original_CT_orientation': list(nib.aff2axcodes(ct_nib.affine)),
            'coordinate_convention': 'NIfTI (i,j,k) -> model stack (k,j,i); no flips',
            'missing_regions': missing, 'region_voxel_counts': counts,
            'qc': qc,
            'status': 'review_required',
            'note': 'Prompts are atlas-derived and are not expert-validated anatomy. '
                    'Never use MRI reference masks for CT-only prompt creation.',
        }
        prompt_obj = {
            'patient_id': pid, 'mode': 'A_atlas_CT_only',
            'ct_path': str(ct_path), 'warped_atlas': 'warped_atlas_on_ct.nii.gz',
            'slice_axis': 'original NIfTI k, model frame z=k',
            'model_frame_order': 'k=0..K-1, exactly as Task 6',
            'box_coordinate_space': 'original CT voxel indices (i=x,j=y), INCLUSIVE extrema',
            'box_usage': 'Pass original image-height/width to MedSAM2 NPZ predictor; '
                         'do NOT scale boxes to 512 yourself if predictor normalizes internally',
            'image_shape_zyx': list(map(int, mask_zyx.shape)),
            'regions': prompts, 'missing_region_ids': missing,
            'status': 'review_required',
        }
        (stage/'prompts.json').write_text(json.dumps(prompt_obj, indent=2, allow_nan=False), encoding='utf-8')
        (stage/'qc_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        if final.exists():
            if final.is_dir() and args.overwrite:
                shutil.rmtree(final)
            elif final.is_dir() and not any(final.iterdir()):
                final.rmdir()
            else:
                raise FileExistsError(f'Existing output cannot be replaced: {final}')
        stage.rename(final)
        print(f'[REVIEW REQUIRED] {pid}: {len(prompts)}/12 regions, {len(qc["flags"])} QC flags; saved {final}', flush=True)
        return final, {'patient_id': pid, 'relative_folder': pid, 'status': 'review_required',
                       'region_count': len(prompts), 'missing_regions': missing,
                       'qc_flag_count': len(qc['flags'])}
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    ap.add_argument('--atlas', type=Path, default=DEFAULT_ATLAS)
    ap.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--task6-root', type=Path, default=ROOT/'data'/'ct_multiwindow')
    choice = ap.add_mutually_exclusive_group()
    choice.add_argument('--patient', type=str, help='Patient ID, e.g. train_040')
    choice.add_argument('--all-development', action='store_true')
    choice.add_argument('--list-development', action='store_true')
    ap.add_argument('--allow-reserved', action='store_true',
                    help='Only for explicitly selected evaluation patients once parameters are frozen')
    ap.add_argument('--registration', choices=['Affine','SyN'], default='Affine',
                    help='Default Affine (linear). SyN is optional nonlinear refinement after QC')
    ap.add_argument('--pad', type=int, default=3, help='Box padding on key slice (native CT pixels)')
    ap.add_argument('--min-voxels', type=int, default=10, help='Review flag threshold for tiny region')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--verbose', action='store_true', help='ANTs registration progress')
    args = ap.parse_args()
    if args.pad < 0 or args.min_voxels < 1:
        ap.error('--pad must be >=0 and --min-voxels >=1')
    document = read_manifest(args.manifest.resolve())
    if args.list_development:
        for p in sorted(document['patients'], key=lambda x:x['patient_id']):
            if p.get('split') == 'development':
                print(p['patient_id'])
        return 0
    choices = select_patients(document, args.patient, args.all_development, args.allow_reserved)
    _ = validate_master(args.template.resolve(), args.atlas.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    print(f'[Task 7] {len(choices)} patient(s) | MNI->CT {args.registration} | CT only; MRI not used', flush=True)
    for patient in choices:
        try:
            out, result = run_patient(patient, args.template.resolve(), args.atlas.resolve(),
                                      output_root, args)
            results.append(result)
        except Exception as e:
            failures.append({'patient_id': patient['patient_id'], 'error': f'{type(e).__name__}: {e}'})
            print(f'[FAILED] {patient["patient_id"]}: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
            if args.verbose:
                traceback.print_exc()
    # Index only this invocation's successful runs; do not claim coverage of all cases.
    invocation_summary = {
        'task': 7, 'created_utc': now(), 'registration': args.registration,
        'patient_count_requested': len(choices), 'completed': results, 'failures': failures,
        'note': 'Review required. Success means a file was produced, NOT an accurate registration.',
    }
    summary_path = output_root/'latest_run_summary.json'
    summary_path.write_text(json.dumps(invocation_summary, indent=2), encoding='utf-8')
    # Cross-patient index for later Task 8 runners; sourced exclusively from
    # the on-disk CT-atlas prompts, never from Task 5 MRI references.
    indexed = {}
    for candidate in sorted(output_root.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith('.'):
            continue
        entry = candidate / 'prompts.json'
        if entry.is_file():
            data = json.loads(entry.read_text(encoding='utf-8'))
            if data.get('mode') == 'A_atlas_CT_only' and data.get('patient_id') == candidate.name:
                indexed[candidate.name] = {
                    'prompts_json': str(entry),
                    'warped_atlas_on_ct': str(candidate/'warped_atlas_on_ct.nii.gz'),
                    'regions': data.get('regions', {}),
                    'status': data.get('status', 'review_required'),
                }
    index_path = output_root.parent / 'modeA_atlas_prompts.json'
    index_path.write_text(json.dumps({
        'task': 7, 'mode': 'A_atlas_CT_only', 'created_utc': now(),
        'source': 'CT + MNI atlas ONLY; never MRI-derived labels',
        'patients': indexed,
    }, indent=2, allow_nan=False), encoding='utf-8')
    print(f'[INDEX] {len(indexed)} CT-atlas prompt records: {index_path}')
    print(f'\n[SUMMARY] {len(results)}/{len(choices)} cases produced | '
          f'{len(failures)} failed | {summary_path}')
    for row in results:
        print(f"  {row['patient_id']}: {row['region_count']}/12 prompts; "
              f"{row['qc_flag_count']} QC flags, {row['status']}")
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as e:
        print(f'[ERROR] {type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(1)
