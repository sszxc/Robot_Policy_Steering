#!/usr/bin/env bash
# Ten LIBERO-10 tasks, ten episodes each, at most two A100 jobs at a time.
#SBATCH --job-name=libero10-array
#SBATCH --account=grp_hbenamor
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --array=0-9%2
#SBATCH --export=NONE
#SBATCH --chdir=/home/xzhan742/Coding/thesis
#SBATCH --output=/scratch/xzhan742/thesis/logs/%x-%A_%a.out
#SBATCH --error=/scratch/xzhan742/thesis/logs/%x-%A_%a.err

set -euo pipefail
source "${THESIS_REPO_ROOT:-$PWD}/platforms/hpc/runtime.sh"
bash platforms/hpc/slurm/smolvla_libero/eval.sbatch task
