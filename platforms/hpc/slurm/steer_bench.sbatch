#!/usr/bin/env bash
# Throughput of PROCSxB layouts on one A100 (t0, random W, 4 batches per process, 120-step episodes).
# sbatch steer_bench.sbatch 1x16 1x32 1x48 2x24 2x32
#SBATCH --job-name=steer-bench
#SBATCH --account=grp_hbenamor
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gpus=a100:1
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --chdir=/home/xzhan742/Coding/thesis
#SBATCH --output=/scratch/xzhan742/thesis/logs/%x-%j.out
#SBATCH --error=/scratch/xzhan742/thesis/logs/%x-%j.err

set -euo pipefail
source "${THESIS_REPO_ROOT:-$PWD}/platforms/hpc/runtime.sh"
libero_runtime
export TMPDIR=$(mktemp -d /tmp/steer-XXXXXX)  # vector-env forkserver sockets fail on scratch
python -m research.pretrained.prepare_policy
out="$THESIS_OUTPUT_ROOT/smolvla/steer2/bench-${SLURM_JOB_ID:?Run using sbatch}"
mkdir -p "$out"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 5 > "$out/gpu.csv" &
smi=$!
(while :; do echo "$(date +%s) $(( $(cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_$SLURM_JOB_ID/memory.usage_in_bytes) >> 20 ))"; sleep 5; done) > "$out/ram.csv" &
ram=$!
for cfg in "$@"; do
  P=${cfg%x*} B=${cfg#*x}
  echo "$(date +%s) start $cfg" >> "$out/events.txt"
  pids=()
  for ((i = 0; i < P; i++)); do
    python -m research.run --policy "$THESIS_DATA_ROOT/models/smolvla-libero-local" --task-id 0 \
      --w random --alpha 0.3 --n-z 8 --scales 1 --zero-reps 0 --inits $((B / 2)) --init-start $((i * B / 2)) \
      --batch-size "$B" --episode-length 120 --videos 0 --output-dir "$out/$cfg-$i" 2>&1 \
      | while read -r l; do echo "$(date +%s) $l"; done > "$out/$cfg-$i.log" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  echo "$(date +%s) end $cfg" >> "$out/events.txt"
done
kill $smi $ram
cat "$out/events.txt"
