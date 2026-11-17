#!/usr/bin/env bash
# One steering run on one A100; all arguments go to research.run unchanged, e.g.
# sbatch platforms/hpc/slurm/steer.sbatch --task-id 9 --batch-size 32 --n-z 15 --inits 4 --alpha 0.5
#SBATCH --job-name=steer
#SBATCH --account=grp_hbenamor
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --chdir=/home/xzhan742/Coding/thesis
#SBATCH --output=/scratch/xzhan742/thesis/logs/%x-%j.out
#SBATCH --error=/scratch/xzhan742/thesis/logs/%x-%j.err

set -euo pipefail
source "${RPS_REPO_ROOT:-$PWD}/platforms/hpc/runtime.sh"
libero_runtime
hostname
nvidia-smi
python -m research.pretrained.prepare_policy
python platforms/hpc/checks/gpu_preflight.py
python -m research.run --policy "$RPS_DATA_ROOT/models/smolvla-libero-local" \
  --output-dir "$RPS_OUTPUT_ROOT/steer-${SLURM_JOB_ID:?Run using sbatch}" "$@"
