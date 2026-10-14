#!/bin/bash
# t2 (ceiling task) on the laptop; resumable
cd "$(dirname "$0")/../../.."
export LIBERO_CONFIG_PATH=$PWD/out/config/libero MUJOCO_GL=egl HF_HUB_OFFLINE=1
mkdir -p out/ticket-local
P=/home/lab/miniforge3/envs/vla-libero/bin/python
$P -m research.ticket --task-id 2 --inject expert_film --dim 8 --zero --batch-size 8 --output-dir out/ticket-local/t2/film8 >> out/ticket-local/t2-film8.log 2>&1
$P -m research.ticket --task-id 2 --inject noise --dim 8 --box 3 --cem-sigma 1 --batch-size 8 --output-dir out/ticket-local/t2/noise8 >> out/ticket-local/t2-noise8.log 2>&1
$P -m research.ticket --task-id 2 --inject noise --dim 1600 --box 3 --cem-sigma 1 --batch-size 8 --output-dir out/ticket-local/t2/noise1600 >> out/ticket-local/t2-noise1600.log 2>&1
