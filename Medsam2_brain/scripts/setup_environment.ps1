$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.python'
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    uv venv --python 3.11 --seed .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
& .venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& .venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints.txt hydra-core iopath tqdm psutil scikit-image
if ($LASTEXITCODE -ne 0) { throw 'Imaging dependency installation failed' }
$env:SAM2_BUILD_CUDA = '0'
& .venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps ../pipeline_medsam2/MedSAM2
if ($LASTEXITCODE -ne 0) { throw 'MedSAM2 installation failed' }
& .venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& .venv\Scripts\python.exe scripts/validate_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed' }
