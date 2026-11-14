#!/bin/bash
# Laptop queue (run detached: setsid nohup bash experiments/libero/film_ablation/local.sh &); re-run to continue.
# FiLM ablation, split W with γ on x̂ (instance), t0/t7, axis grid ±0.5/1/2 on inits 0-7, held-out (inits 10-39 × 2)
# of the best γ and best β pick. R′ = layers 7,15,23,31, d8 (film_split grids reused). α = 0.3·√(4/#layers).
# Stage 0 open loop; 1: d16 (new axes) + spread {31}, {15,31}, 16 CA; 2: front/mid/late × 4; 3: × 1, 2; 4: 2-D γ×β combo.
cd "$(dirname "$0")/../../.."
export LIBERO_CONFIG_PATH=$PWD/out/config/libero MUJOCO_GL=egl HF_HUB_OFFLINE=1
P=/home/lab/miniforge3/envs/vla-libero/bin/python
E=experiments/libero/film_ablation; G=out/libero/smolvla/film_ablation; F=out/libero/smolvla/film_split
mkdir -p $G/logs
FILM="--inject expert_film --w split --center token --batch-size 12"
R=7,15,23,31
declare -A LAYERS=([f1]=3 [f2]=3,5 [f4]=1,3,5,7 [m1]=15 [m2]=15,17 [m4]=13,15,17,19
                   [l1]=27 [l2]=27,29 [l4]=25,27,29,31 [s1]=31 [s2]=15,31 [s16]=$(seq -s, 1 2 31))
declare -A ALPHA=([1]=0.6 [2]=0.4243 [4]=0.3 [16]=0.15)
a_of() { local n; n=$(echo "$1" | tr ',' '\n' | wc -l); echo ${ALPHA[$n]}; }
[ -f $G/axes8.npy ] || $P -c "import numpy as np; np.save('$G/axes8.npy', np.eye(8) * 8**0.5)"
[ -f $G/axes16new.npy ] || $P -c "import numpy as np; np.save('$G/axes16new.npy', np.eye(16)[[4,5,6,7,12,13,14,15]] * 4)"
log() { echo "$(date '+%F %T') $*" >> $G/logs/queue.log; }

steer() {  # name task args...: research.run into a fresh $G/name; DONE marks completion
  local n=$1 d=$G/$1 t=$2; shift 2
  [ -f $d/DONE ] && return; rm -rf $d; log "start $n"
  $P -m research.run $FILM --task-id $t --output-dir $d "$@" >> $G/logs/$n.log 2>&1 && touch $d/DONE
  log "end $n $([ -f $d/DONE ] && echo ok || echo FAILED)"
}
openloop() {  # name task layers dim alpha
  local d=$G/openloop-$1; [ -f $d/openloop.npz ] && return; rm -rf $d
  $P -m research.openloop --task-id $2 --inputs out/libero/smolvla/steer/scan/t$2/inputs.pt --inject expert_film \
    --w split --center token --layers $3 --dim $4 --alpha $5 --axes --n-z $4 --scales 0.25,0.5,1,2,4 --fd 1.0 \
    --noise-reps 8 --max-inputs 96 --batch-size 32 --output-dir $d >> $G/logs/openloop-$1.log 2>&1
}
grid() {  # variant task: 8-axis d8 grid of a layer variant
  local L=${LAYERS[$1]}
  steer grid-t$2-$1 $2 --layers $L --dim 8 --alpha $(a_of $L) --z-file $G/axes8.npy --scales=-2,-1,-0.5,0.5,1,2 --inits 8
}
held() {  # name task tag layers dim alpha pick-args...: pick, then held-out of that z, one run per (tag, pick key)
  local name=$1 t=$2 tag=$3 L=$4 dim=$5 a=$6; shift 6
  local p=$G/pick-t$t-$name.npy key
  key=$($P $E/pick.py $p $dim "$@" 2>> $G/logs/pick.log) || { log "pick FAILED $name t$t"; return; }
  echo "$name held-t$t-$tag-$key" >> $G/picks-t$t.txt  # append log; the last line per name counts
  steer held-t$t-$tag-$key $t --layers $L --dim $dim --alpha $a --z-file $p --scales 1 \
    --zero-reps 0 --z-reps 2 --init-start 10 --inits 30 --videos 0
}
hold_variant() {  # variant task
  local L=${LAYERS[$1]}
  for k in gamma beta; do held $1-$k $2 $1 $L 8 $(a_of $L) $k - $G/grid-t$2-$1; done
}

# Stage 0: open loop (gain per axis; recorded only)
for t in 0 7; do
  openloop t$t-d16 $t $R 16 0.3
  for v in s1 s2 s16 f4 m4 l4 f1 f2 m1 m2 l1 l2; do openloop t$t-$v $t ${LAYERS[$v]} 8 $(a_of ${LAYERS[$v]}); done
done
log "stage 0 done"

# Stage 1: d16 new axes on R′, spread layer counts; d subsets (d1 = d2 picks, d4, d8, d16) and spread held-outs
for t in 0 7; do
  steer grid-t$t-d16 $t --layers $R --dim 16 --alpha 0.3 --z-file $G/axes16new.npy --scales=-2,-1,-0.5,0.5,1,2 --inits 8
  for v in s1 s2 s16; do grid $v $t; done
done
for t in 0 7; do
  for sub in "d1 0 8" "d4 0,1 8,9" "d8 0,1,2,3 8,9,10,11" "d16 - -"; do
    set -- $sub
    held $1-gamma $t R $R 16 0.3 gamma $2 $F/grid-t$t-split-in $G/grid-t$t-d16
    held $1-beta $t R $R 16 0.3 beta $3 $F/grid-t$t-split $G/grid-t$t-d16
  done
  for v in s1 s2 s16; do hold_variant $v $t; done
done
log "stage 1 done"

# Stage 2/3: front / mid / late bands with 4, then 1 and 2 layers
for vs in "f4 m4 l4" "f1 f2 m1 m2 l1 l2"; do
  for t in 0 7; do for v in $vs; do grid $v $t; done; done
  for t in 0 7; do for v in $vs; do hold_variant $v $t; done; done
done
log "stage 2-3 done"

# Stage 4: 2-D grid in the plane of the best γ and best β axis of the d16 picks, a, b ∈ {±0.5, ±1}
for t in 0 7; do
  c=$G/combo-t$t.npy
  [ -f $c ] || $P -c "
import numpy as np
g, b = (int(np.abs(np.load('$G/pick-t$t-d16-' + k + '.npy')[0]).argmax()) for k in ('gamma', 'beta'))
Z = []
for x in (-1, -0.5, 0.5, 1):
    for y in (-1, -0.5, 0.5, 1):
        z = np.zeros(16); z[g], z[b] = 4 * x, 4 * y; Z.append(z)
np.save('$c', np.array(Z)); print('combo axes', g, b)" >> $G/logs/pick.log
  steer grid-t$t-combo $t --layers $R --dim 16 --alpha 0.3 --z-file $c --scales 1 --inits 8
  held combo $t R $R 16 0.3 raw $G/grid-t$t-combo
done
log "film_ablation done"
