#!/usr/bin/env bash
# Ten independent Object tasks share ONE allocated A100; not a Slurm array.
# sbatch experiments/libero/object_parallel/parallel10.sbatch
# Optional: sbatch experiments/libero/object_parallel/parallel10.sbatch EPISODES BATCH_SIZE
# BATCH_SIZE is per task. Default: ten concurrent environments in total.
#SBATCH --job-name=object-parallel
#SBATCH --account=grp_hbenamor
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=a100:1
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --chdir=/home/xzhan742/Coding/thesis
#SBATCH --output=/scratch/xzhan742/thesis/logs/%x-%j.out
#SBATCH --error=/scratch/xzhan742/thesis/logs/%x-%j.err

set -euo pipefail
source "${THESIS_REPO_ROOT:-$PWD}/platforms/hpc/runtime.sh"
export EPISODES="${1:-10}" BATCH_SIZE="${2:-1}" SEED=1000
[[ "$EPISODES" =~ ^[1-9][0-9]*$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || exit 2
(( BATCH_SIZE <= EPISODES && EPISODES % BATCH_SIZE == 0 )) || {
  echo 'BATCH_SIZE must divide EPISODES exactly.' >&2; exit 2;
}
module purge
module load mamba/latest
source activate "$THESIS_PYTHON_ENV"
export PYTHONPATH="$THESIS_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$THESIS_DATA_ROOT/cache/huggingface"
export TMPDIR="$THESIS_DATA_ROOT/tmp"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LIBERO_CONFIG_PATH="$THESIS_DATA_ROOT/config/libero"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
  export MUJOCO_EGL_DEVICE_ID="$CUDA_VISIBLE_DEVICES"
fi
RUN_DIR="$THESIS_OUTPUT_ROOT/libero-object-parallel-${SLURM_JOB_ID:?Run using sbatch}"
mkdir -p "$THESIS_OUTPUT_ROOT" "$TMPDIR" "$THESIS_DATA_ROOT/manifests"
mkdir "$RUN_DIR"
mkdir "$RUN_DIR/logs" "$RUN_DIR/resources"
hostname
nvidia-smi > "$RUN_DIR/resources/gpu-start.txt"
cat "$RUN_DIR/resources/gpu-start.txt"
# Prepare shared inputs once, before any task process starts.
python -m research.pretrained.prepare_policy
export POLICY="$THESIS_DATA_ROOT/models/smolvla-libero-local"
python platforms/hpc/checks/gpu_preflight.py
[[ -f "$LIBERO_CONFIG_PATH/config.yaml" ]] || {
  echo "Missing LIBERO config: $LIBERO_CONFIG_PATH/config.yaml" >&2; exit 1;
}

pids=()
monitor_pid=''
cleanup() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python experiments/libero/object_parallel/monitor_resources.py \
  --root-pid "$$" --output-dir "$RUN_DIR/resources" --interval 5 \
  > "$RUN_DIR/logs/monitor.log" 2>&1 &
monitor_pid=$!
start_seconds=$SECONDS
for task_id in {0..9}; do
  # Each process inherits the SAME CUDA_VISIBLE_DEVICES allocation.
  OUTPUT_DIR="$RUN_DIR/task-$task_id" \
    bash experiments/libero/object_parallel/run_task.sh "$task_id" \
    > "$RUN_DIR/logs/task-$task_id.out" 2> "$RUN_DIR/logs/task-$task_id.err" &
  pids+=("$!")
done
printf 'task_id\tpid\texit_code\n' > "$RUN_DIR/task-status.tsv"
failed=0
for task_id in {0..9}; do
  code=0
  wait "${pids[$task_id]}" || code=$?
  printf '%s\t%s\t%s\n' "$task_id" "${pids[$task_id]}" "$code" >> "$RUN_DIR/task-status.tsv"
  if (( code != 0 )); then failed=1; fi
done
# Reaped PIDs must not be killed by the exit trap.
pids=()
elapsed=$((SECONDS - start_seconds))
if ! kill -0 "$monitor_pid" 2>/dev/null; then
  echo 'Resource monitor failed; inspect logs/monitor.log.' >&2
  failed=1
fi
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=''
if (( failed )); then
  echo "One or more tasks/monitor failed; partial results retained in $RUN_DIR" >&2
  exit 1
fi
python experiments/libero/object_parallel/summarize.py \
  "$RUN_DIR" --episodes "$EPISODES" --elapsed "$elapsed"
echo "Evaluation and resource samples: $RUN_DIR"
