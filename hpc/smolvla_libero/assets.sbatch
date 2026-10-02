#!/usr/bin/env bash
# Fetch the exact locally observed LIBERO asset revision to scratch.
#SBATCH --job-name=libero-assets
#SBATCH --account=grp_hbenamor
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --chdir=/home/xzhan742/Coding/thesis
#SBATCH --output=/scratch/xzhan742/thesis/logs/%x-%j.out
#SBATCH --error=/scratch/xzhan742/thesis/logs/%x-%j.err

set -euo pipefail
module purge
module load mamba/latest
source activate /scratch/xzhan742/thesis/envs/smolvla-libero-py312-v3
ROOT=/scratch/xzhan742/thesis
export HF_HOME="$ROOT/cache/huggingface"
export LIBERO_CONFIG_PATH="$ROOT/config/libero"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
ASSETS="$ROOT/cache/libero/assets"
mkdir -p "$ASSETS" "$ROOT/manifests"

python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
baseline = json.loads(Path('hpc/smolvla_libero/local-baseline.json').read_text())
assets = baseline['assets']
path = snapshot_download(
    repo_id=assets['repo'], repo_type=assets['repo_type'],
    revision=assets['revision'],
    local_dir='/scratch/xzhan742/thesis/cache/libero/assets',
)
required = ('articulated_objects', 'stable_scanned_objects', 'turbosquid_objects',
            'stable_hope_objects', 'scenes')
assert all((Path(path) / name).is_dir() for name in required), path
print('LIBERO assets:', path)
PY

# hf-libero hardcodes ~/.cache/libero/assets. The symlink keeps the data on scratch.
mkdir -p /home/xzhan742/.cache/libero
if [[ -e /home/xzhan742/.cache/libero/assets || -L /home/xzhan742/.cache/libero/assets ]]; then
  [[ "$(readlink -f /home/xzhan742/.cache/libero/assets)" == "$ASSETS" ]]
else
  ln -s "$ASSETS" /home/xzhan742/.cache/libero/assets
fi
if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
  mkdir -p "$LIBERO_CONFIG_PATH"
  printf 'n\n' | python -c 'import libero.libero'
fi
python - <<'PY'
from pathlib import Path
import libero.libero
from libero.libero import get_assets_path
assert Path(get_assets_path()).resolve() == Path('/scratch/xzhan742/thesis/cache/libero/assets')
print('LIBERO asset resolution OK')
PY
