#!/usr/bin/env bash
# One isolated Object task; the HPC launcher runs ten copies on one GPU.
set -euo pipefail
TASK_INDEX="${1:?Usage: run_task.sh TASK_ID}"
[[ "$TASK_INDEX" =~ ^[0-9]$ ]] || exit 2
OUTPUT_DIR="${OUTPUT_DIR:?Set a fresh task output directory}"
POLICY="${POLICY:?Set the pinned local policy path}"
EPISODES="${EPISODES:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-1000}"
INIT_STATE_OFFSET="${INIT_STATE_OFFSET:-0}"
cd "${RPS_REPO_ROOT:?}"
provenance_args=(
  --output-dir "$OUTPUT_DIR" --policy "$POLICY" --episodes "$EPISODES"
  --suite libero_object --batch-size "$BATCH_SIZE" --task-ids "[$TASK_INDEX]" --seed "$SEED"
  --init-state-offset "$INIT_STATE_OFFSET"
)
python -m research.evaluation.provenance "${provenance_args[@]}"
lerobot-eval \
  --seed="$SEED" --policy.path="$POLICY" --policy.device=cuda \
  --env.type=libero --env.task=libero_object --env.task_ids="[$TASK_INDEX]" \
  --env.control_mode=relative --env.init_states=true --env.hard_reset=true \
  --env.init_state_offset="$INIT_STATE_OFFSET" \
  --eval.batch_size="$BATCH_SIZE" --eval.use_async_envs=true \
  --eval.n_episodes="$EPISODES" --env.max_parallel_tasks=1 --output_dir="$OUTPUT_DIR"
python -m research.evaluation.verify_eval "$OUTPUT_DIR" "$EPISODES"
