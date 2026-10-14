#!/usr/bin/env bash
# Run each line of FILE as a shell command on one A100, at most PAR at a time, in file order.
# Lines may chain steps with && and use $POLICY. Logs: <FILE dir>/logs-JOB/, incl. gpu.csv, ram.csv (job cgroup MB).
# sbatch steer_lines.sbatch FILE [PAR=1]; in a job array, %a in FILE is replaced by the array index
#SBATCH --job-name=steer-lines
#SBATCH --account=grp_hbenamor
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gpus=a100:1
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --chdir=/home/xzhan742/Coding/thesis
#SBATCH --output=/scratch/xzhan742/thesis/logs/%x-%j.out
#SBATCH --error=/scratch/xzhan742/thesis/logs/%x-%j.err

set -euo pipefail
source "${THESIS_REPO_ROOT:-$PWD}/platforms/hpc/runtime.sh"
libero_runtime
export TMPDIR=$(mktemp -d /tmp/steer-XXXXXX)  # vector-env forkserver sockets fail on scratch
export POLICY="$THESIS_DATA_ROOT/models/smolvla-libero-local"
hostname
python -m research.pretrained.prepare_policy
FILE=$(realpath "${1//%a/${SLURM_ARRAY_TASK_ID:-}}") PAR=${2:-1}
logs="$(dirname "$FILE")/logs-${SLURM_JOB_ID:?Run using sbatch}"
mkdir -p "$logs"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 10 > "$logs/gpu.csv" &
smi=$!
(while :; do echo "$(date +%s) $(( $(cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_$SLURM_JOB_ID/memory.usage_in_bytes) >> 20 ))"; sleep 10; done) > "$logs/ram.csv" &
ram=$!
i=0
pids=()
while read -r line; do
  [[ -z $line || $line == \#* ]] && continue
  while (( $(jobs -rp | wc -l) >= PAR + 2 )); do wait -n || true; done  # +2: the two loggers
  echo "$(date +%s) start $i: $line" >> "$logs/events.txt"
  (rc=0; bash -c "$line" < /dev/null > "$logs/$i.log" 2>&1 || rc=$?; echo $rc > "$logs/$i.exit") &
  pids+=($!)
  i=$((i + 1))
done < "$FILE"
for p in "${pids[@]}"; do wait "$p" || true; done
kill $smi $ram
echo "$(date +%s) all done" >> "$logs/events.txt"
status=0
for f in "$logs"/*.exit; do [[ $(cat "$f") == 0 ]] || { echo "FAILED: ${f%.exit}.log"; status=1; }; done
exit $status
