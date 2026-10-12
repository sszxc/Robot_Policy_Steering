#!/bin/bash
# Local round 2 (4090 laptop): A = held-out re-eval of every chain's best z (32 new seeds), C = text-only LLM
# (--perm, same (init, rep) seeds and permutations as VLM-PERM/CEM on t7, t9) side by side; then A again for C's chains.
# bash experiments/libero/icl/local_round2.sh   (resumable: re-run to continue)
cd "$(dirname "$0")/../../.."
export LIBERO_CONFIG_PATH=$PWD/out/config/libero MUJOCO_GL=egl
P=/home/lab/miniforge3/envs/vla-libero/bin/python
ROOTS="out/libero/smolvla/icl/hpc out/icl-local"
L=out/icl-local/logs; mkdir -p $L

heldout() { for t in "$@"; do $P -m research.heldout --task-id $t --roots $ROOTS --n 32 --batch-size 8 \
    --output-dir out/icl-heldout/t$t >> $L/heldout-t$t.log 2>&1; done; }

heldout 7 9 0 4 &
for ti in "7 6" "9 3"; do set -- $ti
  $P -m research.optimize --repeats 5 --k 8 --budget 30 --batch-size 8 --task-id $1 --inits $2 --method vlm \
    --text-only --perm --output-dir out/icl-local/t$1/llm-perm >> $L/llm-t$1.log 2>&1
done
wait
heldout 7 9
echo done
