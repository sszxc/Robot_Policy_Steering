#!/usr/bin/env bash
# sbatch eval.sbatch smoke; sbatch --time=04:00:00 eval.sbatch full
#SBATCH --job-name=smolvla-eval
#SBATCH --account=grp_hbenamor
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --chdir=/home/xzhan742/Coding/thesis
#SBATCH --output=/scratch/xzhan742/thesis/logs/%x-%j.out
#SBATCH --error=/scratch/xzhan742/thesis/logs/%x-%j.err

set -euo pipefail
MODE="${1:-smoke}"
case "$MODE" in
  smoke) export TASK_IDS='[0]' EPISODES=1; EXPECTED_EPISODES=1 ;;
  full) unset TASK_IDS; export EPISODES=10; EXPECTED_EPISODES=100 ;;
  task)
    TASK_INDEX="${SLURM_ARRAY_TASK_ID:?task mode requires a Slurm array}"
    [[ "$TASK_INDEX" =~ ^[0-9]$ ]]
    export TASK_IDS="[$TASK_INDEX]" EPISODES=10
    EXPECTED_EPISODES=10
    ;;
  *) echo "Unknown evaluation mode: $MODE"; exit 2 ;;
esac

module purge
module load mamba/latest
source activate /scratch/xzhan742/thesis/envs/smolvla-libero-py312-v3
ROOT=/scratch/xzhan742/thesis
export HF_HOME="$ROOT/cache/huggingface"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export TMPDIR="$ROOT/tmp"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LIBERO_CONFIG_PATH="$ROOT/config/libero"
if [[ "$MODE" == task ]]; then
  export OUTPUT_DIR="$ROOT/results/libero10-task${TASK_INDEX}-${SLURM_ARRAY_JOB_ID}_${TASK_INDEX}"
else
  export OUTPUT_DIR="$ROOT/results/libero10-${MODE}-${SLURM_JOB_ID}"
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$TMPDIR" "$ROOT/manifests"
# EGL sees physical GPU indices; robosuite also honors CUDA_VISIBLE_DEVICES.
if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
  export MUJOCO_EGL_DEVICE_ID="$CUDA_VISIBLE_DEVICES"
fi
hostname
nvidia-smi
git rev-parse HEAD
python hpc/smolvla_libero/prepare_policy.py
export POLICY="$ROOT/models/smolvla-libero-local"

# This preflight uses its own process, so EGL state cannot leak into rollout workers.
python hpc/smolvla_libero/gpu_preflight.py
cd lerobot
bash run_libero10_smolvla.sh
cd ..
python hpc/smolvla_libero/verify_eval.py "$OUTPUT_DIR" "$EXPECTED_EPISODES"
