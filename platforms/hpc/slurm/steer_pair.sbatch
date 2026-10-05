#!/usr/bin/env bash
# Random-W and PCA-W steering grids for one task, two processes sharing one A100.
# sbatch steer_pair.sbatch TASK_ID PCA_PT ALPHA_RANDOM ALPHA_PCA [research.run args...]
# The random run also holds the z=0 noise reps (pass --zero-reps); the PCA run skips z=0.
#SBATCH --job-name=steer-pair
#SBATCH --account=grp_hbenamor
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus=a100:1
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --chdir=/home/xzhan742/Coding/thesis
#SBATCH --output=/scratch/xzhan742/thesis/logs/%x-%j.out
#SBATCH --error=/scratch/xzhan742/thesis/logs/%x-%j.err

set -euo pipefail
source "${THESIS_REPO_ROOT:-$PWD}/platforms/hpc/runtime.sh"
libero_runtime
hostname
nvidia-smi
python -m research.pretrained.prepare_policy
TASK=$1 PCA=$2 ALPHA_RANDOM=$3 ALPHA_PCA=$4
shift 4
out="$THESIS_OUTPUT_ROOT/smolvla/steer/pair-${SLURM_JOB_ID:?Run using sbatch}-t$TASK"
mkdir -p "$out"
run=(python -m research.run --policy "$THESIS_DATA_ROOT/models/smolvla-libero-local" --task-id "$TASK" "$@")
"${run[@]}" --w random --alpha "$ALPHA_RANDOM" --output-dir "$out/random" > "$out/random.log" 2>&1 &
p1=$!
"${run[@]}" --w "$PCA" --alpha "$ALPHA_PCA" --zero-reps 0 --output-dir "$out/pca" > "$out/pca.log" 2>&1 &
p2=$!
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 30 > "$out/gpu.csv" &
smi=$!
status=0
wait $p1 || status=1
wait $p2 || status=1
kill $smi
tail -n 8 "$out"/random.log "$out"/pca.log
exit $status
