#!/usr/bin/env bash
# Smoke test: set TASK_IDS='[0]' and EPISODES=1.
# Standard LIBERO-10 protocol: leave TASK_IDS unset and use EPISODES=10.
set -euo pipefail
RPS_REPO_ROOT="${RPS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export RPS_REPO_ROOT
export PYTHONPATH="$RPS_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$RPS_REPO_ROOT"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# Keep generated LIBERO configuration with runtime outputs.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$(python -c 'from platforms.paths import output_path; print(output_path("config/libero"))')}"

if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
  mkdir -p "$LIBERO_CONFIG_PATH"
  printf 'n\n' | python -c 'import libero.libero'
fi

# The official `lerobot/smolvla_libero` checkpoint expects three cameras
# (`camera1`, `camera2`, `camera3`) and a 6-D state. Standard LIBERO exposes
# two cameras and an 8-D state, which match this HuggingFaceVLA revision's
# `image`, `image2`, and 8-D state inputs.
POLICY="${POLICY:-HuggingFaceVLA/smolvla_libero}"
EPISODES="${EPISODES:-10}"
TASK_IDS="${TASK_IDS:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
OUTPUT_DIR="${OUTPUT_DIR:-$(python -c 'from platforms.paths import output_path; print(output_path("libero/smolvla/vanilla"))')/libero10-$RUN_ID}"
SEED="${SEED:-1000}"
EPISODE_LENGTH="${EPISODE_LENGTH:-}"

args=(
  --seed="$SEED"
  --policy.path="$POLICY"
  --policy.device=cuda
  --env.type=libero
  --env.task=libero_10
  --env.control_mode=relative
  --eval.batch_size=1
  --eval.n_episodes="$EPISODES"
  --env.max_parallel_tasks=1
  --output_dir="$OUTPUT_DIR"
)

if [[ -n "$TASK_IDS" ]]; then
  args+=(--env.task_ids="$TASK_IDS")
fi

if [[ -n "$EPISODE_LENGTH" ]]; then
  args+=(--env.episode_length="$EPISODE_LENGTH")
fi

provenance_args=(
  --output-dir "$OUTPUT_DIR" --policy "$POLICY" --episodes "$EPISODES"
  --task-ids "$TASK_IDS" --seed "$SEED" --episode-length "$EPISODE_LENGTH"
)
python -m research.evaluation.provenance "${provenance_args[@]}"
lerobot-eval "${args[@]}"
