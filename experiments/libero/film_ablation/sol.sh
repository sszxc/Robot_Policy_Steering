#!/bin/bash
# FiLM ablation units for one task (Sol steer_lines lines), resumable via DONE markers. Same design as local.sh, but the
# d16 grid covers all 16 axes (d16g γ 0-7, d16b β 8-15; no film_split grids) and z=0 held-out + open-loop inputs run here.
# bash sol.sh SUITE TASK B UNIT...   UNIT: scan zero open grid:V held:V combo (V: d16g d16b f1..s16; held:d = d subsets)
# env: G (output root), PY (python, default python), POLICY
set -u
S=$1 t=$2 B=$3; shift 3
E=experiments/libero/film_ablation; P=${PY:-python}; G=${G:?set G}
BASE="--policy ${POLICY:-HuggingFaceVLA/smolvla_libero} --suite $S --task-id $t"
FILM="--inject expert_film --w split --center token"
R=7,15,23,31
declare -A LAYERS=([f1]=3 [f2]=3,5 [f4]=1,3,5,7 [m1]=15 [m2]=15,17 [m4]=13,15,17,19
                   [l1]=27 [l2]=27,29 [l4]=25,27,29,31 [s1]=31 [s2]=15,31 [s16]=$(seq -s, 1 2 31))
declare -A ALPHA=([1]=0.6 [2]=0.4243 [4]=0.3 [16]=0.15)
a_of() { local n; n=$(echo "$1" | tr ',' '\n' | wc -l); echo ${ALPHA[$n]}; }
mkdir -p $G/logs
mk() { [ -f $G/$1 ] || $P -c "import numpy as np, os; np.save('$G/$1.$$.npy', $2); os.replace('$G/$1.$$.npy', '$G/$1')"; }
mk axes8.npy "np.eye(8) * 8**0.5"; mk axes16g.npy "np.eye(16)[:8] * 4"; mk axes16b.npy "np.eye(16)[8:] * 4"
log() { echo "$(date '+%F %T') $*" >> $G/logs/queue.log; }
fail=0

run() {  # name module args...: one python run into a fresh $G/name; DONE marks completion
  local n=$1 d=$G/$1 m=$2; shift 2
  [ -f $d/DONE ] && return; rm -rf $d; log "start $n"
  $P -m $m $BASE --output-dir $d "$@" >> $G/logs/$n.log 2>&1 && touch $d/DONE
  [ -f $d/DONE ] && log "end $n ok" || { log "end $n FAILED"; fail=1; }
}
steer() { local n=$1; shift; run $n research.run $FILM --batch-size $B "$@"; }
grid() {  # variant: 8-axis grid (d16g / d16b: the 8 γ / β axes of d16 on R′)
  local Z=$G/axes8.npy L dim=8 a
  case $1 in d16g|d16b) Z=$G/axes16${1:3}.npy L=$R dim=16 a=0.3;; *) L=${LAYERS[$1]}; a=$(a_of $L);; esac
  steer grid-t$t-$1 --layers $L --dim $dim --alpha $a --z-file $Z --scales=-2,-1,-0.5,0.5,1,2 --inits 8
}
held() {  # name tag layers dim alpha pick-args...: pick, then held-out of that z, one run per (tag, pick key)
  local name=$1 tag=$2 L=$3 dim=$4 a=$5; shift 5
  local p=$G/pick-t$t-$name.npy key
  key=$($P $E/pick.py $p $dim "$@" 2>> $G/logs/pick.log) || { log "pick FAILED $name t$t"; fail=1; return; }
  echo "$name held-t$t-$tag-$key" >> $G/picks-t$t.txt  # append log; the last line per name counts
  steer held-t$t-$tag-$key --layers $L --dim $dim --alpha $a --z-file $p --scales 1 \
    --zero-reps 0 --z-reps 2 --init-start 10 --inits 30 --videos 0
}
openloop() {  # name layers dim alpha
  [ -f $G/openloop-t$t-$1/DONE ] && return
  run openloop-t$t-$1 research.openloop --inputs $G/scan-t$t/inputs.pt $FILM --layers $2 --dim $3 --alpha $4 --axes \
    --n-z $3 --scales 0.25,0.5,1,2,4 --fd 1.0 --noise-reps 8 --max-inputs 96 --batch-size 32
}

for u in "$@"; do
  case $u in
    scan) run scan-t$t research.fit_pca --batch-size 8 --episodes 8 --obs-every 20 ;;
    zero) steer held-t$t-zero --layers $R --dim 8 --alpha 0.3 --n-z 1 --zero-reps 2 --z-reps 0 \
            --init-start 10 --inits 30 --videos 0 ;;
    open) openloop d16 $R 16 0.3
          for v in s1 s2 s16 f4 m4 l4 f1 f2 m1 m2 l1 l2; do openloop $v ${LAYERS[$v]} 8 $(a_of ${LAYERS[$v]}); done ;;
    grid:*) grid ${u#grid:} ;;
    held:d) for sub in "d1 0 8" "d4 0,1 8,9" "d8 0,1,2,3 8,9,10,11" "d16 - -"; do
              set -- $sub
              held $1-gamma R $R 16 0.3 gamma $2 $G/grid-t$t-d16g
              held $1-beta R $R 16 0.3 beta $3 $G/grid-t$t-d16b
            done ;;
    held:*) v=${u#held:} L=${LAYERS[${u#held:}]}
            for k in gamma beta; do held $v-$k $v $L 8 $(a_of $L) $k - $G/grid-t$t-$v; done ;;
    combo)  # 2-D grid in the plane of the best γ and best β axis of the d16 grids (own picks; held:d may run alongside)
      c=$G/combo-t$t.npy
      for k in gamma beta; do
        [ -f $G/combo-t$t-$k.npy ] || $P $E/pick.py $G/combo-t$t-$k.npy 16 $k - $G/grid-t$t-d16g $G/grid-t$t-d16b \
          > /dev/null 2>> $G/logs/pick.log
      done
      [ -f $c ] || $P -c "
import numpy as np
g, b = (int(np.abs(np.load('$G/combo-t$t-' + k + '.npy')[0]).argmax()) for k in ('gamma', 'beta'))
Z = []
for x in (-1, -0.5, 0.5, 1):
    for y in (-1, -0.5, 0.5, 1):
        z = np.zeros(16); z[g], z[b] = 4 * x, 4 * y; Z.append(z)
np.save('$c', np.array(Z)); print('combo t$t axes', g, b)" >> $G/logs/pick.log
      steer grid-t$t-combo --layers $R --dim 16 --alpha 0.3 --z-file $c --scales 1 --inits 8
      held combo R $R 16 0.3 raw $G/grid-t$t-combo ;;
    *) echo "unknown unit $u"; exit 2 ;;
  esac
done
exit $fail
