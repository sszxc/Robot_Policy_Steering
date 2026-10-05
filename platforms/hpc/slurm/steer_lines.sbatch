#!/usr/bin/env bash
# Run each line of FILE as `research.run <line>` in parallel on one A100 (lines set --output-dir).
# sbatch steer_lines.sbatch FILE
#SBATCH --job-name=steer-lines
#SBATCH --account=grp_hbenamor
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --chdir=/home/xzhan742/Coding/thesis
#SBATCH --output=/scratch/xzhan742/thesis/logs/%x-%j.out
#SBATCH --error=/scratch/xzhan742/thesis/logs/%x-%j.err

set -euo pipefail
source "${THESIS_REPO_ROOT:-$PWD}/platforms/hpc/runtime.sh"
libero_runtime
export TMPDIR=$(mktemp -d /tmp/steer-XXXXXX)  # vector-env forkserver sockets fail on scratch
hostname
python -m research.pretrained.prepare_policy
pids=()
i=0
while read -r line; do
  [[ -z $line ]] && continue
  python -m research.run --policy "$THESIS_DATA_ROOT/models/smolvla-libero-local" $line \
    > "$THESIS_DATA_ROOT/logs/steer-lines-$SLURM_JOB_ID-$i.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
done < "$1"
status=0
for p in "${pids[@]}"; do wait $p || status=1; done
exit $status
