#!/bin/bash
# Laptop queue: coupled (w=random) vs split FiLM-z8 at parameter-space parity (α=0.3, B=12 as ticket); re-run to continue.
# Stage 1: axis grid z = c·√8·e_k on t0/t7 for both W, then held-out (inits 10-39 × 2) of the best axis z per W
# (split: best γ axis and best β axis). Stage 2: ticket CEM arm film8split, 5 tasks × 2 chains (z = 0 held-out from film8).
cd "$(dirname "$0")/../../.."
export LIBERO_CONFIG_PATH=$PWD/out/config/libero MUJOCO_GL=egl HF_HUB_OFFLINE=1
P=/home/lab/miniforge3/envs/vla-libero/bin/python
G=out/libero/smolvla/film_split; T=out/ticket-local; mkdir -p $G/logs $T/logs
FILM="--inject expert_film --dim 8 --alpha 0.3 --batch-size 12"
[ -f $G/axes.npy ] || $P -c "import numpy as np; np.save('$G/axes.npy', np.eye(8) * 8**0.5)"
steer() {  # name task args...: research.run into a fresh $G/name; DONE marks completion
  local d=$G/$1 t=$2; shift 2
  [ -f $d/DONE ] && return; rm -rf $d
  $P -m research.run $FILM --task-id $t --output-dir $d "$@" >> $G/logs/$(basename $d).log 2>&1 && touch $d/DONE
}
for t in 0 7; do
  for w in random split; do steer grid-t$t-$w $t --w $w --z-file $G/axes.npy --scales=-2,-1,-0.5,0.5,1,2 --inits 8; done
done
for t in 0 7; do
  for sel in random:all split:gamma split:beta; do
    w=${sel%:*} s=${sel#*:} b=$G/best-t$t-${sel/:/-}.npy
    [ -f $b ] || $P experiments/libero/film_split/pick.py $G/grid-t$t-$w $b $s >> $G/logs/pick.log
    steer held-t$t-${sel/:/-} $t --w $w --z-file $b --scales 1 --zero-reps 0 --z-reps 2 --init-start 10 --inits 30 --videos 0
  done
done
for t in 0 7 9 4 2; do
  $P -m research.ticket --task-id $t --inject expert_film --dim 8 --w split --repeats 2 --batch-size 12 \
    --output-dir $T/t$t/film8split >> $T/logs/t$t-film8split.log 2>&1
done
echo "film_split done" >> $G/logs/queue.log
