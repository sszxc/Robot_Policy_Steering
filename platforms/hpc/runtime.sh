#!/usr/bin/env bash
# Source from a Slurm script before loading its Python environment.
THESIS_REPO_ROOT="${THESIS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ -f "$THESIS_REPO_ROOT/platforms/hpc/config.local.sh" ]]; then
  source "$THESIS_REPO_ROOT/platforms/hpc/config.local.sh"
fi
export THESIS_REPO_ROOT
export THESIS_PLATFORM=hpc
export THESIS_DATA_ROOT="${THESIS_DATA_ROOT:-/scratch/xzhan742/thesis}"
export THESIS_OUTPUT_ROOT="${THESIS_OUTPUT_ROOT:-$THESIS_DATA_ROOT/results}"
export THESIS_PYTHON_ENV="${THESIS_PYTHON_ENV:-$THESIS_DATA_ROOT/envs/smolvla-libero-py312-v3}"
cd "$THESIS_REPO_ROOT"
