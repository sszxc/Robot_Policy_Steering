#!/usr/bin/env bash
# Smoke test: set TASK_IDS='[0]' and EPISODES=1.
# Standard LIBERO-10 protocol: leave TASK_IDS unset and use EPISODES=10.
set -euo pipefail

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# Keep LIBERO's one-time path configuration inside this checkout rather than ~/.libero.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PWD/.libero}"

if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
  mkdir -p "$LIBERO_CONFIG_PATH"
  printf 'n\n' | python -c 'import libero.libero'
fi

# The `lerobot/smolvla_libero` Hub revision currently declares three cameras,
# while standard LIBERO exposes two. This original release has the matching
# `image`, `image2`, and 8-D state inputs.
POLICY="${POLICY:-HuggingFaceVLA/smolvla_libero}"
EPISODES="${EPISODES:-10}"
TASK_IDS="${TASK_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/libero10_smolvla}"
EPISODE_LENGTH="${EPISODE_LENGTH:-}"

args=(
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

lerobot-eval "${args[@]}"
