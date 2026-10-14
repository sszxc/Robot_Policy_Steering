#!/bin/bash
# Laptop queue (Sol in maintenance 2026-10-12..15): one run at a time, B=12, priority order; resumable (re-run to continue).
# Stage 1: film8 / noise8, 1 repeat, t0 t7 t9 t4 t2. Stage 2: noise1600 t0 t7. Stage 3: second repeat.
cd "$(dirname "$0")/../../.."
export LIBERO_CONFIG_PATH=$PWD/out/config/libero MUJOCO_GL=egl HF_HUB_OFFLINE=1
P=/home/lab/miniforge3/envs/vla-libero/bin/python
O=out/ticket-local; mkdir -p $O/logs
declare -A ARM=([film8]="--inject expert_film --dim 8 --zero" [noise8]="--inject noise --dim 8 --box 3 --cem-sigma 1"
                [noise1600]="--inject noise --dim 1600 --box 3 --cem-sigma 1")
run() { $P -m research.ticket --task-id $1 ${ARM[$2]} --repeats $3 --batch-size 12 --output-dir $O/t$1/$2 >> $O/logs/t$1-$2.log 2>&1; }
for t in 0 7 9 4 2; do run $t film8 1; run $t noise8 1; done
for t in 0 7; do run $t noise1600 1; done
for t in 0 7 9 4 2; do run $t film8 2; run $t noise8 2; done
echo done >> $O/logs/queue.log
