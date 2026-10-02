#!/usr/bin/env bash
# Authorized setup stage. No evaluation or GPU allocation.
#SBATCH --job-name=smolvla-setup
#SBATCH --account=grp_hbenamor
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --chdir=/home/xzhan742/Coding/thesis
#SBATCH --output=/scratch/xzhan742/thesis/logs/%x-%j.out
#SBATCH --error=/scratch/xzhan742/thesis/logs/%x-%j.err

set -euo pipefail
module purge
module load mamba/latest

ROOT=/scratch/xzhan742/thesis
ENV_PREFIX="$ROOT/envs/smolvla-libero-py312-v3"
export CONDA_PKGS_DIRS="$ROOT/cache/mamba"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export HF_HOME="$ROOT/cache/huggingface"
export TMPDIR="$ROOT/tmp"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR" "$ROOT/manifests"

if [[ -e "$ENV_PREFIX" ]]; then
  echo "Environment already exists: $ENV_PREFIX. Stop for inspection."
  exit 1
fi

# Platform libraries are solved by mamba; Python package versions use local constraints.
mamba create --yes --override-channels -c conda-forge -p "$ENV_PREFIX" \
  python=3.12.14 pip 'ffmpeg=8.*' libstdcxx libgcc
source activate "$ENV_PREFIX"

CONSTRAINTS="$PWD/hpc/smolvla_libero/local-constraints.txt"
python -m pip install --constraint "$CONSTRAINTS" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130'
python -m pip install --constraint "$CONSTRAINTS" --editable './lerobot[smolvla,libero]'
python -m pip check
python - <<'PY'
import av
import torch
from torchcodec.decoders import VideoDecoder
print('Video stack import OK:', av.__version__, torch.__version__, VideoDecoder.__name__)
PY

# Download on the allocated CPU compute node; pin both repos to local cache revisions.
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
baseline = json.loads(Path('hpc/smolvla_libero/local-baseline.json').read_text())
downloaded = {}
for key in ('policy', 'backbone'):
    model = baseline[key]
    assert model['revision'], f'Missing revision: {key}'
    downloaded[key] = snapshot_download(repo_id=model['repo'], revision=model['revision'])
Path('/scratch/xzhan742/thesis/manifests/model-paths.json').write_text(
    json.dumps(downloaded, indent=2) + '\n'
)
PY

mamba list --explicit > "$ROOT/manifests/conda-explicit-$SLURM_JOB_ID.txt"
python -m pip freeze > "$ROOT/manifests/pip-freeze-$SLURM_JOB_ID.txt"
git rev-parse HEAD > "$ROOT/manifests/git-head-$SLURM_JOB_ID.txt"
module list 2> "$ROOT/manifests/modules-$SLURM_JOB_ID.txt"
python - <<'PY'
import torch
print('Installed torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('Setup complete. GPU/EGL/LIBERO rollout validation requires a separate GPU job.')
PY
