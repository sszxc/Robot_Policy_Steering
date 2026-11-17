#!/usr/bin/env bash
# Source from a Slurm script before loading its Python environment.
RPS_REPO_ROOT="${RPS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export RPS_REPO_ROOT
export RPS_DATA_ROOT="${RPS_DATA_ROOT:-/scratch/xzhan742/thesis}"
export RPS_OUTPUT_ROOT="${RPS_OUTPUT_ROOT:-$RPS_DATA_ROOT/results}"
export RPS_PYTHON_ENV="${RPS_PYTHON_ENV:-$RPS_DATA_ROOT/envs/smolvla-libero-py312-v3}"
cd "$RPS_REPO_ROOT"

# GPU LIBERO jobs: Python env, offline caches, EGL rendering.
libero_runtime() {
  module purge
  module load mamba/latest
  source activate "$RPS_PYTHON_ENV"
  export PYTHONPATH="$RPS_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export HF_HOME="$RPS_DATA_ROOT/cache/huggingface"
  export TMPDIR="$RPS_DATA_ROOT/tmp"
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export LIBERO_CONFIG_PATH="$RPS_DATA_ROOT/config/libero"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  mkdir -p "$TMPDIR" "$RPS_DATA_ROOT/manifests"
  # EGL sees physical GPU indices; robosuite also honors CUDA_VISIBLE_DEVICES.
  if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
    export MUJOCO_EGL_DEVICE_ID="$CUDA_VISIBLE_DEVICES"
  fi
}
