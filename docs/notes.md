# HX5-D20 + UR5e MuJoCo hand tasks — results and notes

Goal: get a feel for the ROBOTIS HX5-D20 right hand in sim (UR5e arm). Privileged state only (no cameras),
no domain randomization. RL comes later, so collisions are primitives and hand-only scenes exist.

## Layout
- `assets/mujoco/hx5_right.xml` — hand with primitive collisions (source of truth; vendored `robotis_hand/` untouched).
- `assets/mujoco/scene_base.xml`, `assets/mujoco/objects/*.xml` — world and task objects.
- `robotics/sim/scene.py` — `hand_only(pos, quat)`, `arm_hand()`, `add_table`, `add_object`, `hand_quat(palm_dir, finger_dir)`.
- `robotics/sim/runner.py` — `Sim`: `reset`, `move(arm_q, hand_q, duration)`, `ik`/`ik_high`/`move_site`, `cart_move` (straight line), `tip_ik` (fingertips), `hold`, video/viewer/log.
- `robotics/sim/hand.py` — 20-dim hand target layout and named postures.
- `robotics/policies/scripted/*.py` — one script per task (commands in `docs/usage.md`).
- `robotics/sim/view.py` — interactive viewer for base scenes.

Headless rendering needs `MUJOCO_GL=egl`.

## Hand facts (sim model)
- Hand frame: palm normal +x, fingers +z, thumb side +y. Sites: `palm`, `<finger>_tip`.
- 20 joints, all independently position-controlled: kp=4, force limit ±0.387 Nm per joint.
- Base mass is a placeholder (0.4 kg); no adapter between UR5e flange and hand yet.

## Task list and priority
| # | Task | Approach | Status |
|---|------|----------|--------|
| 1 | Multi-pose grasps (power / pinch / tripod / lateral / hook; top & side) | script, arm+hand | done: 5/5 lift 15 cm; payload upper bounds power ~6 kg (suspicious), pinch ~0.64 kg |
| 2 | Spray bottle trigger | script, arm+hand | done: 3/3 full pulls, bottle held |
| 3 | Hammer a nail | script, arm+hand | done: 2.2 cm in 8 strikes, 230 g hammer |
| 4 | Scissors open/close | script, hand-only | done: 5 cycles, 75–85% of range; paper cut faked by weld release |
| 5 | In-hand position alignment (slide/pivot a rod to a target) | script first, RL later | partial: slide −20 mm ±0.4, pivot +16° ±0.2, translate ±6 mm ±1 mm; one direction only, RL needed for general |
| 6 | Cube arbitrary reorientation | RL (hand-only, palm up) | done (self-collision, 2026-09-25): MJX+PPO ~154M steps (36.7M + 117M warm-start in out/rl/cube/cont); eval 4.0 consecutive successes (0.4 rad tol) per 30 s, 0 drops (5 ep); 5.0 / 1 drop over 20 ep |
| 7 | Screwdriver twirl with fingers only | scripted + RL | RL done (self-collision, 2026-09-25): 281–288 deg/s CPU (~290 MJX), both directions, thumb/index/middle gaiting; scripted: index-only 32 deg/s one direction |
| 8 | Pen spinning (yaw about palm normal) | RL | partial (self-collision retrain, 2026-09-25): one half-turn (~185°) then holds, 0.17 rad/s both directions; 0/10 CPU, 0/64 MJX drops; exploit checks passed. Pre-fix ~5 rad/s relied on finger pass-through; 6 runs (out/rl/pen/run1–6, best run6 54.1M) did not recover gaiting |
| 9 | Baoding balls | RL (scripted failed) | done (self-collision, 2026-09-25): ~21 half-swaps per 20 s (CPU, 1/10 drops; MJX 66.2 rad, 9% drops); fast flicking gait, ignores 16 s target pace; pace-tracking variant 442°/450° per 20 s, 0 drops |
| 10 | Rubik's cube one-face turn | scripted (fixture) done; in-hand RL not solved | done (fixture): 4 consecutive quarter turns, 1 thumb stroke each; in-hand RL 0.25 turns/ep, see report |

## Decisions log
- 2026-09-23: replaced the mesh-collision combined model (`ur5e_hand.xml`) with MjSpec composition in `robotics/sim/scene.py`.
- 2026-09-23: own DLS IK instead of adding `mink` (6-DoF arm, one site target is enough).
- 2026-09-23: installed into base conda: pytest, mujoco-mjx 3.7.0, jax[cuda12] 0.11.2, brax 0.14.2 (RL uses MJX + brax PPO on GPU).
- 2026-09-23 consolidation: scene_base has `compiler angle="radian"` (spec-added joint ranges were silently degrees)
  and multiccd (flat cylinder ends; MJX accepts the flag); `arm_hand(base_height)` adds a pedestal; wrist servos kv 100 → 25
  (`scene.WRIST_KV`); ARM_HOME is now the wrist-up branch; Sim gained `cart_move` (lerp + slerp), `ik_high`
  (multi-seed, wrist above site, nearest to current ctrl) and `tip_ik`; postures CURL/PINCH/TRIPOD/KEY_PINCH/HOOK/POWER in sim/hand.py.
- 2026-09-24: RL envs had no hand self-collision (hand masks 0/1 → fingers passed through each other by up to 25 mm in
  all RL policies). Fix: `scene.HAND_SELF_PAIRS` (57 explicit frictionless pairs, sampled so every >1 mm overlap config
  has one in contact) added by `scene.add_hand_self_pairs` in every RL env; MJX cost ~1.4x (bitmask all-pairs 14x).
  Scripted tasks were unaffected (default masks). RL results in the task table predate this and need retraining.
- Arm position servos sag ~1 cm under the hand's weight (no gravity compensation); tasks close the loop on object pose where it matters.

## Hand limits seen
- Fingers can't wrap thin handles alone (35.5 mm phalanges, 20 mm thick): objects must seat against the palm.
- Thumb opposes only in front of the palm; it can't face the index pad squarely (~55° off at yaw limit −1.571).
  Joint ranges match the vendored ROBOTIS model, so this is kept as-is.
- Adjacent straight fingers touch after ~0.05 rad inward MCP roll (index→middle); flexed 1.6 rad, ~0.15 rad.
  The vendored mesh model gives the same onsets within ~0.02 rad, so roll is mainly for spreading outward.
- Straight-thumb CMC barely moves the tip sideways; grip hold/slip threshold is ~0.01 rad of extra squeeze.

## Task #10 report — Rubik's cube one-face turn (2026-09-23)
**Scripted, cube in a table fixture (`robotics/policies/scripted/rubik.py`): works.** Arm brings the palm down over a fixed cube; only the thumb
sweeps the top layer (CMC −1.4 → 1.0, retract outside), closed-loop on the hinge angle, plus a 4-fold detent (0.01 Nm peak).
4 consecutive quarter turns, one stroke each: −89.4 / −178.4 / −267.3 / −357.0°. Robustness: 24/27 cube offsets within ±6 mm pass
(6/27 without the detent, so the snap-to-quarter behaviour matters). Scripted in-hand (palm up) failed: one turn at most, very sensitive.

**RL, cube free in the palm-up hand (`robotics/policies/rl/rubik_env.py`, `rubik_train.py`, `rubik_eval.py`): not solved.**
- Obs 76 (hand qpos/qvel/ctrl, cube pose/vel, top-layer error sin/cos, hinge vel). Reward: 20·progress + 0.5/(err+0.1)
  − 10·cube displacement − 0.5·cube rotation + 100·success − 100·drop. Success: err < 10° and hinge vel < 1 rad/s, then target += 90°.
- PPO 39M steps (9000 s timeout of a 50M run). Training-eval peak 1.18 quarter turns/ep at 25M, drop ≈ 0.55 throughout.
- Re-evaluated (64 eps in MJX, 20 eps in CPU MuJoCo):

| checkpoint | MJX turns / drop | CPU turns / drop |
|---|---|---|
| 25.2M (best in log) | 0.08 / 0.36 | 0.50 / 0.00 |
| 39.0M (final) | 0.25 / 0.47 | 0.00 / 1.00 (flings cube in ~5 steps) |

- Takeaways: the policy never learned a reliable regrasp gait (holding the lower layers with ring/pinky + palm while
  index/thumb push the top is at the edge of this hand's workspace); results are high-variance and the MJX→CPU gap is large
  for this contact-rich task. Next: fixture-based RL first (lower layers fixed), then a curriculum loosening the fixture;
  longer training; check MJX vs CPU contact settings (solver iterations, condim) before trusting CPU evals.
- 2026-09-25 retrain with hand self-collision: still unsolved. Best run-1 39.0M (out/rl/rubik/best_params.pkl): MJX 0.07 turns / 0.48 drop (256 ep), CPU 0.50 / 0.00 (one +1 turn only; −1 ~0). Tried `--assist_prob 0.5 --w_vel 2` (spring fixture curriculum + anti-fling) and `--w_prog 50 --detent 0`: flat. Even with the cube held the policies never find the turn stroke → next: scripted/demo prior (thumb stroke from robotics/policies/scripted/rubik.py). Rank checkpoints with `python -m robotics.policies.rl.rubik_mjx_eval --params ...`.
- RL screwdriver: `python -m robotics.policies.rl.screwdriver_train` (stopped at 68M steps, 3.5 h on shared GPU; plateaued from ~52M; best_params = params_68157440), eval `python -m robotics.policies.rl.screwdriver_eval` → out/rl/screwdriver/screwdriver.mp4. RL model uses a capsule handle (MJX lacks cylinder–box). Caveats: fingertips penetrate up to 4.8 mm (soft contact, 5 solver iters); thumb/index/middle run at the 0.387 Nm limit.
- RL baoding: two-stage PPO (14.4M steps holding at 8 s/rev, then ~52.8M at 16 s/rev with progress weight 50, split 18.35M + 34.4M resume in stage2b/); tracks the pace until ~20M stage-2 steps, fast flicking from ~23M. Best stage2b/params_32112640 → out/rl/baoding/best_params.pkl, eval `python -m robotics.policies.rl.baoding_eval` → out/rl/baoding/baoding.mp4; pace-tracking 18.35M → pace_params.pkl (`--params out/rl/baoding/pace_params.pkl --video out/rl/baoding/baoding_pace.mp4`). MJX sweep in out/rl/baoding/mjx_eval.txt. Scripted CEM gait failed (144° in 20 s): curled fingertips only reach the finger-side half of the palm, so nothing pushes balls toward the wrist.
- RL pen: pen rests on the proximal phalanges of slightly curled fingers (flat on the palm it can't be pushed sideways → first 110M-step run learned to hold still). Reward 2·yaw rate (clip ±3 rad/s), small offset/tilt terms, drop −10; NaN-safe obs. 204M steps (~2.1 h), best 183.5M → out/rl/pen/best_params.pkl; eval `python -m robotics.policies.rl.pen_eval`, exploit checks `python -m robotics.policies.rl.pen_check` → out/rl/pen/pen.mp4. Index/middle/ring drive CCW, ring/pinky CW; thumb unused; pen also rolls about its own axis while yawing.

## Baoding: noisy teacher → vision policy (plan 2026-10-31; sim only, no real robot)
Goal: pace-tracking Baoding from a top-down RGB camera. Two students from the same noisy env: DAgger distillation
of a noisy privileged teacher, and direct vision PPO (asymmetric critic). Code: `robotics/policies/rl/baoding_noisy_*`,
`baoding_render.py`, `baoding_dagger.py`.
- Step 0 (`baoding_check.py`): both old checkpoints are closed-loop: ball obs frozen at reset → pair turn 64.6 → 0.7
  rad/ep (best), 7.6 → 0.8 (pace); a 0.15 m/s ball kick at 10 s changes nothing.
- Reward: tracking `Σ exp(-err/15mm)` + `w_turn · clip(Δpair, -0.3, ω·dt)` (progress capped at the target pace, so
  flicking faster earns nothing) − action² − 50·drop. Period 16 s, one direction.
- Noise (actor side): ball pos σ 2 mm/step + per-episode bias σ 1 mm, last 3 ball positions instead of ball velocity,
  joint pos/vel noise, action noise σ 0.05, per-episode 0/1-step action delay, random ball kicks (~1%/step), random
  start phase (balls placed on the orbit, target starts aligned). No physics randomization (no sim2real goal).
- Critic sees the clean 78-dim privileged obs (+ delay flag, ball bias).
- Camera: fixed top-down (along −palm normal), 64×64 RGB, 2 stacked frames; pure-JAX ray caster over the collision
  primitives (spheres/capsules/boxes) so it batches inside MJX. Balls red/blue (identity matters for the targets).
- Vision actor sees pixels + hand proprio + target phase only (no ball state).
- Results (MJX, 128 eps × 20 s, noisy env unless [clean]; target 7.85 rad = 450°; `baoding_noisy_eval`):

| policy | pair turn (rad/ep) | drop | track err | pixels frozen at reset |
|---|---|---|---|---|
| teacher, 49.0M steps (`teacher/best_params.pkl`, 72 min) | 7.88 (100%) | 0.00 | 2.6 mm | — |
| teacher [clean] | 7.89 | 0.00 | 2.6 mm | — |
| DAgger student, 10 iters × 192 eps (`dagger/student.pkl`, 12 min) | 7.87 (100%) | 0.00 | 2.6 mm | 2.42, drop 0.33 |
| DAgger student [clean] | 7.87 | 0.00 | 2.6 mm | 2.40, drop 0.38 |
| vision PPO, 48.0M steps (`vision_rl/best_params.pkl`, 2.5 h) | 7.80 (99%) | 0.01 | 3.5 mm | 0.68, drop 0.43 |
| vision PPO [clean] | 7.70 | 0.02 | 3.9 mm | 0.70, drop 0.38 |

  Teacher reached the pace by ~11M steps and plateaued from ~23M. DAgger: plain BC (iter 1 rollouts) already 7.63 rad /
  2% drops; later iters close the gap. Both vision policies collapse with frozen pixels, so they read the balls from the image.
  Vision PPO (512 envs, ~6.7k steps/s): pace by ~2M steps, plateau from ~15M (99% pace, 1% drops, 4.8 mm);
  checkpoints 14.6M–60.6M all 98–101% pace, 0–3% drops, 3.6–4.8 mm. Repeat evals of one checkpoint vary
  ~±0.05 rad / ±1% drop (GPU nondeterminism). Verdict: both routes work in sim; teacher + DAgger took 84 min
  (72 + 12) vs 2.5 h for vision PPO and matches the teacher exactly; vision PPO tracks ~1 mm worse.
  Caveats: images are flat-shaded collision primitives from a fixed camera, no image noise/augmentation,
  and the target phase is given to the policy; not a sim2real result.

## Baoding v2: speed command + MuJoCo Warp (2026-11-03; sim only)
Same noisy env, plus a per-episode speed command and real mesh rendering. Flags: `--periods 2 16 --warp`
(`baoding_noisy_train`, `baoding_dagger`, `baoding_noisy_eval --cmd`); outputs in `out/rl/baoding_cmd/`.
- Command: period T log-uniform in [2, 16] s/rev, ω = 2π/T, one direction; ω/π is the last entry of every obs
  (state, vis_state, privileged_state); reward cap = ω·dt. No curriculum was needed: the teacher learned all of [2, 16].
- MuJoCo Warp (MJX `impl="warp"`; needs `mujoco-warp==3.7.0` + `warp-lang==1.12.0`, 1.19 breaks the MJX import):
  batched rendering, meshes and lights work in jit/vmap. At 1024 envs: physics 88k env-steps/s (vs 15k for MJX-JAX),
  64 px render ~130 µs/img (mesh face count dominates). Chosen: 64 px, full hand meshes, no shadows, camera "top"
  matching the old ray-caster view. All v2 policies train and evaluate on Warp physics (the v1 teacher transfers to it
  unchanged). `naconmax` is a total over worlds (64 per world); contact buffers are shared across worlds, so episode
  re-sampling only touches qpos/qvel/ctrl.
- Vision PPO run 1 (no over-speed penalty) reward-hacked: it spun faster than commanded at every speed (pace 172–187%,
  34 mm, stopped at 10M; `vision_rl_overspeed/`). Fix `--w_over 1`: subtract `w_turn·clip(Δpair − ω·dt, 0, 0.3)`.
  The teacher and DAgger used w_over 0, so the vision PPO reward differs slightly.
- Jobs were run one at a time: a second GPU job next to vision PPO (0.7 mem fraction) crashed it with
  `CUDA_ERROR_ILLEGAL_ADDRESS`.
- Results (`baoding_noisy_eval --cmd`, 128 eps × 20 s per cell, noisy env; pace = achieved / commanded pair rotation,
  drops count as no progress; switch = command changed at 10 s, pace over 5–10 s / 10–13 s / 13–20 s):

| policy | T=2 s | T=3 s | T=4 s | T=8 s | T=16 s | switch 8→3 s | switch 3→8 s |
|---|---|---|---|---|---|---|---|
| teacher, 83.1M steps (17 min) | 100%, 30.5 mm | 101%, 16.6 | 101%, 11.7 | 100%, 5.9 | 101%, 5.6 | 100 / 103 / 101% | 100 / 88 / 97% |
| DAgger, 10 iters × 192 eps (9 min) | 99%, 30.5 mm | 101%, 16.9 | 101%, 11.9 | 100%, 5.9 | 100%, 6.0 | 99 / 102 / 100% | 100 / 88 / 98% |
| DAgger, pixels frozen | 2% | 3% | 4% | 8% | 19% | — | — |
| vision PPO, 60.6M steps, w_over 1 (4.4 h) | 72%, 34.0 mm | 86%, 32.6 | 90%, 30.4 | 98%, 9.3 | 98%, 7.0 | 99 / 76 / 86% | 88 / 127 / 93% |
| vision PPO, pixels frozen | 1% | 1% | 1% | 3% | 6% | — | — |

  Drops: teacher and DAgger 0.00 everywhere; vision PPO ≤ 0.02; frozen 0.07–0.31. Sweep videos (16→8→4→2 s/rev,
  MuJoCo view | 64 px policy view): `teacher_sweep.mp4` (99% pace), `dagger_sweep.mp4` (98%), `vision_rl_sweep.mp4` (80%).
  Vision PPO plateaued at 88–91% training-eval pace from ~25M steps; checkpoints all look alike, so the last one is used.
- Verdict: teacher + DAgger (26 min total) follow the command over the whole range and match each other exactly;
  vision PPO tracks slow commands (T ≥ 8 s) but lags at T ≤ 4 s and overshoots briefly after slowing down.
  Both students collapse with frozen pixels, so they read the balls from the image.
- Caveats: track error grows with speed for every policy (the balls lag the target phase; the reward checks speed,
  not phase). Fixed camera, no image noise/augmentation, target phase given to the policy; not a sim2real result.

## Baoding v3: toward the real hand — wrist camera, palm tilt, physics DR (2026-11-04; sim only)
Goal (user): make the vision policy transferable in principle: wrist camera close to the wrist base with an oblique
view, a large palm-tilt range with the tilt as a policy input, physics DR. No real robot / SysID yet. Stages are added
one at a time, each with its own conclusion. Route: noisy privileged teacher → DAgger RGB student (vision PPO skipped:
4× the cost and worse in v2). All runs local (4090 laptop). Outputs `out/rl/baoding_{wrist,tilt,dr}/`.
Env flags (`baoding_noisy_env.BaodingNoisyEnv`, also on train / dagger / eval): `--cam wrist`, `--tilt DEG`, `--dr`.

### Stage A: oblique wrist camera
- Camera (`cam="wrist"`): aimed at the orbit center from the wrist side, optical axis 55° off the palm normal, 15 cm
  away (≈13 cm above the palm, 3.7 cm behind the hand-base origin), fovy 42° = square center crop of the D435 color
  stream (the lab RealSense is a D435). Thumb lower right. During Baoding the curled fingers and the near ball
  partly occlude the far ball (top camera: never).
- Same v2 teacher (physics unchanged); DAgger 10 iters × 192 eps (10 min), `--periods 2 16 --warp --cam wrist`.
- Results (`baoding_noisy_eval --cmd --cam wrist`, 128 eps × 20 s, noisy env):

| policy | T=2 s | T=3 s | T=4 s | T=8 s | T=16 s | switch 8→3 s | switch 3→8 s |
|---|---|---|---|---|---|---|---|
| DAgger wrist cam | 100%, 29.7 mm | 101%, 17.1 | 101%, 11.8 | 100%, 5.8 | 100%, 5.8 | 100 / 103 / 100% | 100 / 88 / 97% |
| DAgger wrist cam, pixels frozen | 2% | 3% | 4% | 7% | 14% | — | — |
| (v2 DAgger top cam, for reference) | 99%, 30.5 | 101%, 16.9 | 101%, 11.9 | 100%, 5.9 | 100%, 6.0 | 99 / 102 / 100% | 100 / 88 / 98% |

  No drops (frozen: 0.10–0.31).
- Conclusion A: the oblique, partly occluded wrist view costs nothing: the student matches the teacher (and the
  top-camera student) at every speed and still reads the balls from the image (frozen pixels → collapse).
  Two stacked frames + the ball-position aux head are enough to handle the occlusion. Video `baoding_wrist/dagger_sweep.mp4`.

### Stage B: palm tilt (gravity direction) as a policy input
- Tilt = angle between "up" and the palm normal, azimuth = which side is downhill (0 fingers, 90 thumb, 180 wrist,
  270 pinky); nominal v1/v2 tilt is 17° fingers-downhill. `--tilt DEG`: per episode, uniform over the spherical cap
  ≤ DEG. Applied as a per-body force m·(g_new − g_nom) at each CoM (hand base welded, camera fixed to the hand);
  identical to rotating `opt.gravity` (CPU check: max diff 0.0). Obs: unit gravity in the hand frame (3) before the
  command in state / vis_state / privileged_state; actor copies carry a per-episode σ 0.03 error (≈2°, FK + mount).
- B0, zero-shot v2 teacher (no gravity input; `eval --cmd --tilt_grid --tilt 45 --no_g_obs`, 64 eps, pace% (drop)):

| tilt | T=4 s fingers / thumb / wrist / pinky downhill | T=8 s fingers / thumb / wrist / pinky |
|---|---|---|
| 0° (flat) | 4 (0.95) | 12 (0.94) |
| 10° | 101 (0.00) / 7 (0.83) / 2 (1.00) / −1 (1.00) | 99 (0.00) / 13 (0.77) / 4 (1.00) / −2 (1.00) |
| 17° | 101 (0.00) / 6 (0.77) / 2 (1.00) / −1 (1.00) | 100 (0.00) / 16 (0.72) / 4 (1.00) / −3 (1.00) |
| 25° | 100 (0.00) / 5 (0.72) / 2 (1.00) / −2 (1.00) | 100 (0.00) / 15 (0.66) / 4 (1.00) / −3 (1.00) |
| 35° | 88 (0.02) / 4 (0.73) / 2 (1.00) / −2 (1.00) | 100 (0.00) / 10 (0.80) / 3 (1.00) / −3 (1.00) |
| 45° | 72 (0.03) / 3 (0.92) / 2 (1.00) / −2 (1.00) | 100 (0.00) / 8 (0.86) / 3 (1.00) / −3 (1.00) |

  Conclusion B0: the v2 gait only works with the fingers downhill (the balls rest against the fingers); even a flat
  palm drops 95% of episodes, and any thumb/wrist/pinky-downhill tilt drops almost always. Tilt robustness has to be
  trained, the current policy has none.
- B1 teacher, cap ≤45° about palm-up, all azimuths (`--tilt 45`, w_over 0, 200M steps, 34 min,
  `baoding_tilt/teacher_up45/`): training eval plateaued at ~50% drops from ~100M steps. Tilt grid at T=6 s:

| tilt | fingers | thumb | wrist | pinky downhill |
|---|---|---|---|---|
| 0° | 54 (0.14) | | | |
| 15° | 101 (0.00) | 173 (0.02) | 3 (1.00) | 4 (0.22) |
| 30° | 101 (0.00) | 176 (0.00) | 3 (1.00) | −1 (0.78) |
| 45° | 99 (0.00) | 150 (0.14) | 2 (1.00) | −2 (0.92) |

  Fingers-downhill works at every tilt; thumb-downhill holds the balls but over-spins (no over-speed penalty);
  pinky-downhill only holds at 15° and wrist-downhill always drops (the balls roll off the palm heel, where nothing
  can stop them). So a palm-up-centered cap wastes samples on (probably) infeasible directions.
- B2: cap ≤35° about the nominal 17° fingers-downhill tilt (`--tilt_about nominal`), `--w_over 1`, 300M steps.
  Angle of the grid cells from nominal (in the cap if ≤35°): 0°→17; 15° fingers/thumb/wrist/pinky → 2/23/32/23;
  30° → 13/34/47/34; 45° → 28/48/62/48.
- B2 teacher (`--tilt 35 --tilt_about nominal --w_over 1`, 300M steps, 50 min, `baoding_tilt/teacher/`): training
  eval 93% pace / 12–20% drops at the end (still creeping up). Tilt grid (64 eps; * = outside the 35° cap):

| tilt | T=3 s fingers / thumb / wrist / pinky | T=8 s fingers / thumb / wrist / pinky |
|---|---|---|
| 0° | 82 (0.06) | 99 (0.00) |
| 15° | 100 (0.00) / 77 (0.02) / 8 (0.89) / 75 (0.20) | 100 (0.00) / 97 (0.03) / 14 (0.88) / 77 (0.20) |
| 30° | 100 (0.00) / 75 (0.02) / 1* (1.00) / 25 (0.72) | 100 (0.00) / 98 (0.02) / 3* (1.00) / 27 (0.67) |
| 45° | 99 (0.00) / 54* (0.27) / 1* (1.00) / 7* (0.91) | 100 (0.00) / 77* (0.23) / 3* (1.00) / 5* (0.91) |
- B2 DAgger wrist-cam student (`baoding_dagger --cam wrist --tilt 35 --tilt_about nominal`, 10 iters, 10 min,
  `baoding_tilt/dagger/`). Tilt grid (64 eps):

| tilt | T=3 s fingers / thumb / wrist / pinky | T=8 s fingers / thumb / wrist / pinky |
|---|---|---|
| 0° | 86 (0.03) | 99 (0.00) |
| 15° | 100 (0.00) / 77 (0.02) / 9 (0.84) / 73 (0.22) | 100 (0.00) / 98 (0.02) / 18 (0.83) / 77 (0.19) |
| 30° | 100 (0.00) / 73 (0.02) / 1* (1.00) / 26 (0.69) | 100 (0.00) / 95 (0.03) / 3* (1.00) / 25 (0.69) |
| 45° | 100 (0.00) / 56* (0.23) / 1* (1.00) / 7* (0.89) | 100 (0.00) / 79* (0.19) / 3* (1.00) / 6* (0.89) |

  Random training-like episodes (`eval --random`, 2 × 128 eps, speed T∈[2,16] s, tilt in the cap): teacher 88/78%
  pace, 0.09/0.18 drops; student 88/76%, 0.08/0.20.
- Conclusion B: tilt has to be an input and has to be trained (v2: no robustness at all). With the gravity direction
  as input, one policy covers fingers-downhill (any tilt ≤45°), flat palm and thumb-downhill ≤30° at ~100% (slow) /
  75–100% (T=3 s). Pinky-downhill is only partly solved (≈75% at 15°, ≈25% at 30°) and wrist-downhill is not solved
  at all (the balls roll off the palm heel). These are likely limits of this hand (no wall at the wrist/pinky side),
  not of the distillation: the wrist-camera student matches the teacher cell by cell. Practical rule for the real
  arm: keep the palm within ~30° of "fingers 17° downhill" and avoid tilting toward the wrist or pinky.

### Stage C: physics DR (+ tilt)
- `--dr` (per episode): ball mass ×[0.7,1.3] (each ball), hand friction ×[0.5,1.2] (hand–ball contacts use the hand's
  1.0), ball friction ×[0.5,1.5] (ball–ball), per-joint kp ×[0.7,1.3] (kv ×√, damping ratio kept), per-dof damping
  and armature ×[0.5,2] log-uniform, frictionloss [0,0.01] Nm, finger-link masses ×[0.8,1.2], ball contact solref
  timeconst [0.01,0.03] s, action delay 0/1/2 steps (p 0.4/0.4/0.2). Implemented by overriding model fields per
  episode inside the vmapped step (MJX-Warp batches them); tilt forces use the randomized masses. Critic gets a
  7-dim DR summary. NaN states only appear after an episode has already ended (drop), never inside one.
- Teacher (`--tilt 35 --tilt_about nominal --w_over 1 --dr`, 400M steps, 75 min, `baoding_dr/teacher/`): training
  eval plateaued at ~80% pace / 20–26% drops from ~200M. DAgger wrist-cam student 10 iters (10 min).
- Tilt grid under DR (64 eps), teacher | student:

| tilt | T=8 s teacher: fingers / thumb / wrist / pinky | T=8 s student | T=3 s teacher | T=3 s student |
|---|---|---|---|---|
| 0° | 62 (0.00) | 60 (0.03) | 42 (0.03) | 37 (0.05) |
| 15° | 100 / 99 / 4 (0.95) / 2 (0.45) | 100 / 97 / 4 (0.98) / 2 (0.52) | 99 / 66 / 1 / 1 (0.47) | 98 / 65 / 1 / 1 (0.52) |
| 30° | 100 / 100 / 4* / −2 (0.86) | 100 / 94 / 4* / −2 (0.83) | 99 / 63 / 1* / −1 (0.83) | 100 / 61 / 1* / −1 (0.89) |
| 45° | 100 / 77* (0.23) / 3* / −4* | 100 / 65* (0.36) / 3* / −3* | 99 / 45* / 1* / −1* | 99 / 44* / 1* / −1* |

- Conclusion C: the DR teacher keeps fingers- and thumb-downhill but gives up the pinky side (B: ~75% at 15°, now
  ~2%) and the flat palm drops to ~60% (B: 99%). The student still copies the teacher exactly. In the cross matrix
  below the B student (never saw DR) is *better* under DR physics than the C policies (76% vs 68%), so these DR
  ranges cost little at test time but made the teacher's training harder / narrower at 400M steps. Not yet a
  positive result for DR; next: warm-start the DR teacher from B2 (actor dims match; pad the critic input) or a DR
  curriculum, and narrow the friction range once the real balls/finger pads are known.

### Stage D: visual DR
- `--vis_dr` (DAgger only; the teacher ignores pixels): camera-mount jitter ±1.5 cm per axis + ≤5° rotation (applied
  to the camera pose the renderer reads from data), ball color gain ×[0.75,1.15] per channel, hand gray [0.1,0.6]
  ± tint, floor texture tint ×[0.3,1.2], then contrast ×[0.8,1.2], brightness ±0.1, white balance ×[0.9,1.1], pixel
  noise σ∈[0,0.03]. Example views: `baoding_vis/vis_grid.png`. Student = C teacher, `--dr --vis_dr`, 10 iters (10 min);
  its imitation loss stays higher (action MSE 0.0067 vs 0.0045 for C).

### Cross matrix (all wrist-cam students; `eval --random`, 2 × 128 training-like episodes per cell, speed T∈[2,16] s)
Rows = policy (stage it was trained for), columns = test conditions; pace% / drop rate, mean of the two batches.
"nominal" = fixed 17° fingers-downhill, nominal physics, clean images; "tilt" = 35° cap about nominal.

| policy | nominal | tilt | tilt + DR | tilt + DR + visual DR |
|---|---|---|---|---|
| A student (wrist cam, no tilt input) | 101 / 0.00 | 20 / 0.63 | 18 / 0.67 | 5 / 0.63 |
| B student (+ tilt input) | 99 / 0.00 | 82 / 0.14 | 76 / 0.20 | 9 / 0.55 |
| C student (+ physics DR) | 99 / 0.00 | 72 / 0.16 | 68 / 0.19 | 7 / 0.56 |
| D student (+ visual DR) | 98 / 0.00 | 68 / 0.18 | 62 / 0.21 | 59 / 0.25 |
| C teacher (state, for reference) | 99 / 0.00 | 72 / 0.16 | 68 / 0.19 | 68 / 0.18 |

No episode ended in a NaN state. Track error grows 7–10 mm (nominal) → 12–17 mm (tilt/DR).

### Overall conclusions (v3)
1. Wrist camera (oblique, occluded): free. The student matches the teacher at all speeds.
2. Palm tilt: the single biggest gap. Without the tilt input/training, performance outside "fingers downhill"
   collapses (A: 20% pace, 63% drops in the cap). With it, one policy covers fingers/thumb-downhill and the flat
   palm; wrist- and pinky-downhill are probably beyond this hand. Keep the real palm in that region.
3. Physics DR (these ranges) costs a trained-without-DR policy only ~6 points at test time, while DR training at
   400M steps gave a narrower teacher; DR needs a better training recipe before it pays off.
4. Visual DR is mandatory for any image change: B/C students collapse (5–9% pace) under camera/color/photometric
   randomization, the D student keeps 59% (87% of its teacher's 68%) and loses only ~4 points on clean images.
5. DAgger stays lossless whenever the images are clean (student = teacher in every stage); the remaining
   student–teacher gap (68 → 59) comes from the visual randomization.

### Real camera on the printed mount (2026-11-10; estimate from CAD, not measured)
- Camera: ELP-USBGS1200P01-H120 (AR0234 global shutter, 1920×1200, 3 µm, 38×38 mm board, M12 "low distortion"
  lens, HFOV 120° per spec table / ~126° per listing text). Pinhole at HFOV 120°: f ≈ 554 px, VFOV 94.5°.
- `out/print/Robotis Hand - UR Arm Mount.stl` holds three parts in the lab `hand_mount.stl` frame: the hand–flange
  adapter (same surfaces as the lab mesh), a RealSense bracket (D4xx plate, 2×M3 45 mm apart) and an ELP bracket
  (X plate, 34.0 mm square hole pattern). Both brackets have the same 3.5 mm base plate that drops into the
  adapter's top slot (y 7–10.5) on the PCD-63 bolts at 60/90/120°; the ELP one is exported 30 mm off along +y.
  Mounted ELP bracket = part shifted by (0, −30, 0): `out/print/mount_views/elp_bracket_mounted.stl`.
  Mesh → hand frame: hb = (mz, mx, my − 18) mm (lab `ur5.xml`: hand_mount quat 0.5⁴, hand_attach 24 mm).
- Assuming the board lies on the X face that points at the fingers and the optical center is 15 mm in front of
  it (±5 mm, unknown lens height): camera at hb (46, −48, 11) mm (25 mm above the palm ≈ ball-center height, pinky
  side, just in front of the hand base), optical axis hb (0, 0.326, 0.946) = parallel to the palm, yawed 19° toward
  the thumb. tool0 frame: (−48, −46, 35) mm, axis (0.326, 0, 0.946). Orbit center 89 mm away, 14° off-axis.
  The camera cannot go higher (user): only the palm-plane side is available.
- Consequence: the orbit plane is seen edge-on (balls move left–right and change size; the near ball hides most of
  the far one twice per turn), unlike the 55° oblique sim view. Renders: `out/print/mount_views/` (elp_views,
  policy64, third_person).
- Student encoder option `baoding_dagger --enc ssm` (per-frame CNN, 32×32 maps → 32 spatial-softmax keypoints per
  frame; 0.33M params vs 2.4M flatten); not trained yet (GPU held by the film_split queue).

### Wrist camera calibrated; sim camera replaced (2026-11-12)
- Calibration (ros2_ur `scripts/calib`, ChArUco on a laptop screen, 81 views): intrinsics rms 0.35 px, rational
  model, HFOV 109.6° / VFOV 91.0° at 640×480 (f ≈ 240 px); hand-eye reprojection 0.78 px, board spread 1.3 mm /
  0.40°. Copy of the result: `assets/mujoco/lab/wrist_camera.json` (K, dist, `T_hand_cam`, `T_tool0_cam`).
- Camera in the hand frame: (44.3, −47.3, 18.5) mm, optical axis (−0.010, 0.223, 0.975) = parallel to the palm,
  yawed 13° toward the thumb (CAD estimate: (46, −48, 11) mm, 19°); image down = palm normal (hand body at the top), thumb left.
- Sim: camera `wrist` on `hand_mount` in `lab/ur5.xml` = ideal pinhole on the calibrated optical axis, fovy 90°,
  88×64 px → 108°×90°. This lies inside the real image (−56..54° × −45..46° about the axis) and holds the balls
  over the whole orbit (−53..9° × −21..18°; a 90° square would cut the thumb-side ball).
- `baoding_noisy_env` `cam="wrist"` now reads that camera (pose, fovy, resolution) instead of the 55° oblique
  placeholder; the wrist image is 88×64 (top stays 64×64), `baoding_dagger` takes the size from the env. Earlier
  wrist-cam students / results (stages A–D above) used the old oblique 64×64 view and do not load into this one.
- Real image → policy image (lens distortion removed, same pinhole as the sim):
  `Kn = [[32, 0, 43.5], [0, 32, 31.5], [0, 0, 1]]`;
  `m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, Kn, (88, 64), cv2.CV_32FC1)`; `cv2.remap(img, m1, m2, ...)`
  (blur or area-average first: one policy pixel covers about 5–7 real pixels).
- Checked: lab scene rendered from `wrist` at recorded arm + finger angles overlays the undistorted real frames
  (fingers and forearm); Warp env returns (64, 88, 6) for wrist and (64, 64, 6) for top. Not retrained yet.

## Baoding v4: toward the real hand (2026-11-12)
Arm static during the rotation (per-episode tilt, 35° cap about nominal as in v3). Real data from ros2_ur
`out/calib`: `balls/` (33 frames, clicked circles, colors), `background/` (115 frames around the palm-up arm pose).

### Calibration check
- Two independent calibrations (11-11: 26 views, 11-12: 81 views, different screens) agree: f 242.4 / 240.6 px,
  camera pose 1.8 / 1.1 / 3.2 mm and 0.56°. `ur5.xml` camera = `wrist_camera.json` (position, rotation).
- Weak spot: thumb-side image edge (few corners there). Same ray projected with both intrinsics: 1–3 px in the
  center / right, 7 px at −45°, 10 px at −53° (the thumb-side ball's extreme) = up to ~1.3 policy px. Covered by the
  camera jitter below; no recapture.

### Stage 0: env aligned to the real setup
- Balls: 10 g (`baoding.py` MASS), white = ball0, yellow = ball1 (albedo from the measured sRGB). The 60 g B2 teacher
  zero-shot at 10 g: 92 / 81% pace, 0.09 / 0.17 drops (`eval --random`, 2 × 128; at 60 g it was 88 / 78%, 0.09 / 0.18),
  so the lighter balls do not change the task; later teachers warm-start from B2.
- Wrist image: rendered at 2× (176×128), floor not rendered, rays that miss (depth 0) = background mask, composite,
  then 2×2 average to 88×64. One fill light along the view direction (the real balls have no dark side).
- `--vis_dr` v2 (`VIS` in `baoding_noisy_env.py`), per episode: background = 70% a real lab frame (`--bg`, library from
  `baoding_real --bg`; rotation ±40°, zoom 1–1.6, shift ±25%, flip, mirrored borders) / 30% smooth random colors,
  gain ×[0.6, 1.3]; camera jitter ±5 mm, ≤3° (was ±15 mm, 5°: CAD guess → calibrated); light directions rotated ≤40°;
  ball gain ×[0.75, 1.15] with ±7% tint; hand gray [0.03, 0.25] (real ≈ 0.1); image delay 1 (p 0.7) or 2 steps (real
  image age at a 20 Hz policy ≈ 35–70 ms); contrast / brightness / white balance ±7% / pixel noise as before.
- Not randomized: focal length (the renderer precomputes its rays; ±2% ≈ 1 px at the edge, inside the rotation jitter).
- Images: `docs/results/baoding_real_vis.html` (real frames, sim before / after the composite, backgrounds alone).
  Ball pixels, mean sRGB real vs sim: white (225, 223, 203) vs (207, 213, 203); yellow (205, 184, 67) vs (217, 209, 62).
- Known differences left: the real palm / mount fills more of the image top than the sim hand base; the real
  background frames contain the (open) real fingers, which stay in the composite as distractors.
- Two `BaodingNoisyEnv(warp=True)` in one process fail in `mjx.forward` (model metadata equality): one env per process.

### Stage 1: teachers at 10 g (`out/rl/baoding_real/`, chain `pipeline.sh`)
- Teacher: B2 warm start, `--tilt 35 --tilt_about nominal --w_over 1`, 150M steps (28 min). DR teacher: warm start
  from it (`--dr`, ball mass ×[0.9, 1.1], other ranges as v3; critic input padded), 200M steps (39 min).
- Tilt grid, pace% (drop), 64 eps, fingers / thumb / wrist / pinky downhill (DR teacher under DR physics):

| tilt | teacher T=3 s | teacher T=8 s | DR teacher T=3 s | DR teacher T=8 s |
|---|---|---|---|---|
| 0° | 100 (0.00) | 100 (0.00) | 96 (0.03) | 100 (0.00) |
| 15° | 100 / 95 / 18 (0.81) / 82 (0.16) | 100 / 98 / 21 (0.81) / 90 (0.09) | 100 / 95 / 14 (0.86) / 70 (0.28) | 100 / 99 / 16 (0.86) / 69 (0.28) |
| 30° | 100 / 91 / 1 (1.00) / 37 (0.61) | 100 / 99 / 3 (1.00) / 41 (0.56) | 100 / 87 / 1 (1.00) / 25 (0.73) | 100 / 99 / 3 (1.00) / 28 (0.69) |
| 45° | 100 / 74 (0.22) / 1 / 9 (0.89) | 100 / 81 (0.20) / 3 / 9 (0.88) | 99 / 68 (0.28) / 1 / 12 (0.86) | 100 / 76 (0.27) / 3 / 7 (0.89) |

- Conclusion 1: the 10 g teacher is better than v3 B2 at T=3 s (flat palm 82 → 100, thumb-downhill 77 → 95). The
  warm-started DR teacher keeps the flat palm (v3 C from scratch: 42–62%, now 96–100%) and most of the pinky side
  (v3 C: ~2% at 15°, now 70%), so warm start fixes the "DR makes the teacher narrower" problem of v3. Wrist-downhill
  stays unsolved, as expected.

### Stage 2: wrist-camera students (DR teacher, `--dr --vis_dr --bg`, 10 iters, 128 envs; 18–33 min each)
`eval --random`, 2 × 128 episodes per cell, pace% / drop rate (mean of the two batches), T∈[2,16] s:

| policy | nominal | tilt | tilt + DR | tilt + DR + visual DR |
|---|---|---|---|---|
| DR teacher (state) | 100 / 0.00 | 87 / 0.13 | 85 / 0.14 | 85 / 0.15 |
| flat, 2 frames | 99 / 0.01 | 79 / 0.17 | 76 / 0.22 | 69 / 0.27 |
| flat, 3 frames | 98 / 0.02 | 82 / 0.16 | 77 / 0.21 | 70 / 0.26 |
| ssm, 2 frames | 100 / 0.00 | 81 / 0.16 | 79 / 0.20 | 77 / 0.21 |
| ssm, 3 frames | 100 / 0.00 | 84 / 0.15 | 79 / 0.20 | 78 / 0.20 |

- Conclusion 2: the spatial-softmax encoder is the better student for the edge-on real view: it loses 1–2 points
  under the visual randomization (flat: 7) and reaches 92% of its teacher (78 vs 85) with background, delay and
  lighting randomized (v3 D student: 59 vs 68 = 87%, with weaker randomization). A third frame adds ~1 point. Use
  `dagger_ssm3` (ssm2 is equivalent within noise).

### Stage 3: open-loop check on real frames (`baoding_real --check`): weak evidence only
- Ball-position head on the 33 labelled real frames, projected into the 88×64 image, vs the clicked ball centers
  (ball radius ≈ 12 px). Median error white / yellow, fraction within 8 px:

| student | proprio = sim default pose | proprio = real joints |
|---|---|---|
| flat2 | 4.8 / 2.3 px, 0.76 / 0.94 | 4.6 / 5.7 px, 1.00 / 0.91 |
| ssm2 | 4.5 / 3.9 px, 0.88 / 0.85 | 4.6 / 7.9 px, 0.97 / 0.52 |
| flat3 | 5.5 / 2.3 px, 0.79 / 0.79 | 6.8 / 7.6 px, 0.58 / 0.58 |
| ssm3 | 2.9 / 3.1 px, 0.82 / 1.00 | 6.6 / 27 px, 0.64 / 0.00 |
| constant (orbit center) | 5.0 / 9.0 px, 0.67 / 0.42 | |

- With in-distribution proprio the students put both balls on the real balls (ssm3: 2.9 / 3.1 px, clearly better
  than the constant guess for the yellow ball), so the real
  images do drive the vision features. With the real joints the head breaks (ssm3 yellow: 27 px): the real frames
  were taken with a nearly open hand (MCP ≈ 0.2 rad, training poses ≈ 1.2 rad), and the head leans on proprio.
- Limits: balls were static and partly held by a human hand; the clicked radius is a color patch, not the ball edge,
  so there is no depth ground truth; the white ball sits near the constant guess in most frames. This check cannot
  confirm closed-loop transfer.
- Next: (1) record real frames with the fingers in the sim start pose (`E.DEFAULT_POSE`) and balls at several orbit
  phases; (2) add a ball-position head on the image features alone (no proprio), so the probe tests vision only;
  (3) then closed loop on the real hand at T = 8–16 s, palm at the nominal tilt.

### Stage 3b: image-only ball head (`dagger_ssm3_img`, 2026-11-12)
- Student gets a second ball-position head on the image features alone (keypoints → 128 → 6, no proprio; same loss
  weight). Same recipe as ssm3 (33 min). Sim, pace% / drop: nominal 100 / 0.01, tilt 83 / 0.16, tilt + DR 79 / 0.19,
  tilt + DR + visual DR 80 / 0.20 (ssm3: 78 / 0.20): the extra head costs nothing.
- Real frames (same 33, no new data): image-only head 3.2 / 3.3 px median (white / yellow), 97% / 100% within 8 px
  (constant guess 5.0 / 9.0 px, 67% / 42%; proprio head with the real open-hand joints 13.9 / 6.6 px). It follows
  the balls when they swap sides and with a human hand in view: `docs/results/baoding_real/ssm3_img_real_check.png`.
- Conclusion 3: the visual randomization transfers at the level of "where are the two balls" on static real frames.
  Not tested: moving balls, real finger poses of the gait, closed loop. Deploy `dagger_ssm3_img/student.pkl`.

## Baoding v5: perturbations, no orbit tracking (2026-11-13; `out/rl/baoding_real/`, chains `pipeline6.sh`, `pipeline7.sh`)
Goal: checkpoints for the first real closed loop. Balls may be placed anywhere on the palm; more physics gaps covered.

### Stage 0: code
- Student: proprio ball head removed (`dagger_ssm3_img2`: nominal 100 / 0.00, tilt + DR + visual DR 78 / 0.20; real
  frames 2.5 / 1.9 px). No cost.
- `PERT` in `baoding_noisy_env.py` (all off by default; `--pert key=value` in train / dagger / eval): ball forces
  (white, sustained push, spin kick), finger-link push, gravity wander, joint zero offset, proprio delay, action
  delay jitter, start noise, backlash, rate limit; model-level: torsional / rolling friction, ball radius / inertia,
  actuator force limit, palm / finger friction. External forces go through `xfrc_applied` (as the tilt does).
- `--rel`: no orbit tracking. Reward = capped pair rotation + flat-bottomed pair-center term (full within 15 mm of
  the palm center); the 6 goal obs entries are zero (same layout: tracking checkpoints warm-start); start = pair
  center offset y ±12 mm, z −20…+4 mm, gap 1–16 mm, any angle (`REL_INIT`).

### Stage 1: sensitivity of the tracking policies (one perturbation at a time, strong level)
DR teacher / `dagger_ssm3_img2`, tilt + DR + visual DR, 2 × 128 episodes, pace% / drop. Repeats differ by ~±2 points.

| perturbation | teacher | student |
|---|---|---|
| none | 84 / 0.14 | 79 / 0.18 |
| white ball force, std 0.5 ball weights | 56 / 0.50 | 44 / 0.62 |
| ball push, std 1.5 weights, 0.3 s | 64 / 0.46 | 58 / 0.52 |
| joint target rate limit 1 rad/s (nominal max 3) | 56 / 0.32 | 47 / 0.42 |
| joint zero offset, std 0.1 rad | 73 / 0.24 | 52 / 0.41 |
| backlash 0.1 rad | 78 / 0.18 | 69 / 0.26 |
| actuator force limit × 0.5 | 76 / 0.17 | 68 / 0.24 |
| ball radius × 0.95 / × 1.05 | 81 / 0.17, 82 / 0.15 | 71 / 0.25, 77 / 0.20 |
| gravity wander, std 0.15 rad | 82 / 0.17 | 74 / 0.26 |
| proprio one step old | 83 / 0.14 | 76 / 0.21 |
| finger friction × 0.5 | 83 / 0.17 | 76 / 0.26 |
| finger-link push 2 N, ball spin 30 rad/s, action-delay jitter 0.5, palm friction × 0.5 | no effect | no effect |
| rolling friction 2 mm (condim 6) | 88 / 0.12 | 82 / 0.17 |

- Conclusion 1: forces on the balls, joint speed limit and joint zero offset are the large ones (the student is hit
  much harder than the teacher by the zero offset). Rolling friction helps slightly; not checked that condim 6 is
  active in MJX-Warp. Rate / force limits and ball radius are model-level constants here: not trained on, they need
  the real hand's numbers.

### Stage 2: teacher A (`teacher_rel`: `--rel --dr`, warm start from the DR teacher, 200M steps, 38 min)
- Training eval flat from the start (~79% pace while alive, drop 0.23–0.30). Diagnostic, 512 episodes, tilt + DR:

| | pace% / drop |
|---|---|
| DR teacher, tracking, orbit start | 84 / 0.16 |
| teacher A, orbit-like start | 81 / 0.15 |
| teacher A, wide start | 71 / 0.24 |

- Wide-start drops happen in the first 0.5 s (median step 6) and follow the tilt: gravity fingers-downhill component
  > 0.3: 0.04–0.08; wrist-downhill: 0.83. Wrist-side starts (0.32) and large gaps (0.29) add a little.
- Conclusion 2: the tracking reward is not needed. The wide start costs ~9 points of drops, in tilts that were
  already hard; at the real nominal tilt (17° fingers downhill) it is cheap.
- Sim-to-sim warning: both teachers drop 0.53 in MJX-JAX physics on CPU (Warp: 0.16). Cause not investigated.
- Student `dagger_rel` (ssm3): nominal 97 / 0.00; tilt + DR + visual DR 66 / 0.27 (teacher 71 / 0.24); real frames
  4.1 / 2.0 px.

### Stage 3: perturbation-trained variants (P), `pipeline7.sh`
- P (about half the strong levels): white ball force 0.25 weights, push 1.0 weights, joint zero offset 0.05 rad,
  backlash U[0, 0.06] rad, gravity wander 0.08 rad, proprio delay p = 0.5 (delay and backlash are per episode from
  here on; stage 1 used fixed values). Teachers: 200M steps, warm start (`teacher_rel` → `teacher_rel_p`,
  `teacher_dr` → `teacher_trk_p`); ssm3 students trained under the same P.
- Tilt + DR + visual DR, 2 × 128 episodes, teacher | student, pace% / drop. rel rows use the wide start. "trk" =
  DR teacher / `dagger_ssm3_img2` from stage 1 (its backlash row used a fixed 0.1 rad, not comparable).

| perturbation | trk | trk_p | rel | rel_p |
|---|---|---|---|---|
| none | 84 / 0.14, 79 / 0.18 | 82 / 0.15, 74 / 0.21 | 70 / 0.25, 66 / 0.28 | 70 / 0.24, 58 / 0.31 |
| ball force 0.5 | 56 / 0.50, 44 / 0.62 | 61 / 0.38, 54 / 0.45 | 44 / 0.56, 37 / 0.61 | 48 / 0.49, 42 / 0.52 |
| push 1.5 | 64 / 0.46, 58 / 0.52 | 77 / 0.23, 68 / 0.31 | 58 / 0.44, 51 / 0.54 | 65 / 0.32, 53 / 0.40 |
| zero offset 0.1 | 73 / 0.24, 52 / 0.41 | 78 / 0.17, 54 / 0.37 | 62 / 0.29, 47 / 0.41 | 64 / 0.26, 40 / 0.38 |
| backlash ≤ 0.1 | – | 80 / 0.17, 73 / 0.22 | 64 / 0.26, 58 / 0.29 | 64 / 0.25, 52 / 0.32 |
| rate 1 rad/s | 56 / 0.32, 47 / 0.42 | 56 / 0.33, 47 / 0.43 | 41 / 0.41, 38 / 0.42 | 46 / 0.39, 39 / 0.41 |

- Nominal (no tilt, no DR): trk_p 100 / 0.00, 99 / 0.00; rel 98 / 0.00, 97 / 0.00; rel_p 97 / 0.00, 90 / 0.01.
  Real frames, image head: trk_p 3.3 / 2.5 px, rel_p 3.9 / 2.7 px.
- Conclusion 3: P training helps the teachers without a baseline cost (push drops halve for trk_p: 0.46 → 0.23).
  The students keep about half of that gain under ball forces and pay 5 (trk_p) to 8 (rel_p) points without
  perturbations. Joint zero offset stays a student problem (teacher 78, student 54) and the rate limit is untouched
  (not trained on). `dagger_rel_p` is the weakest student.
- Checkpoints for the real hand (`out/rl/baoding_real/<name>/student.pkl`): `dagger_ssm3_img2` (tracking),
  `dagger_trk_p` (tracking + P), `dagger_rel` (no orbit, goal obs = zeros), `dagger_rel_p` (no orbit + P).
  Tracking students need the target phase set from the observed pair angle at the start.
- Open: students under joint zero offset; rate / force limits and ball radius as per-episode DR once the real
  hand's numbers are known; the JAX-vs-Warp gap.

## Baoding v6: thumb engagement, real-hand stuck states (2026-11-14; `out/rl/baoding_real/`, chain `pipeline8.sh`)
Goal (user): a thumb that takes part (mainly looks / human-likeness, secondarily robustness), and a fix for the
stuck states seen on the real hand (route A: train and deploy with the same target dynamics).

### Stage 0: diagnosis from the first real closed loop (2026-11-13, T = 4 s) and code
- Thumb unused already in sim: ball-contact fraction thumb / index / middle / ring / pinky = 0.01 / 0.55 / 0.91 /
  0.91 / 0.63 (DR teacher, T = 4 s, tilt + DR; students 0.02–0.07 for the thumb). Not a reach limit: the thumb can
  touch a ball anywhere on the orbit (11–28% of random thumb poses). Likely cause: with fingers 17° downhill the
  thumb-side arc (wrist → thumb side → fingertips) runs downhill and gravity does the work; the uphill half
  (fingertips → pinky side → wrist) is ring / pinky work. Reversing the direction makes the thumb side uphill.
- Real thumb blocked: with j2 ≈ 0, j1 stops at ≈ −0.48 while commanded −1.57 (23% of those steps; 30 s in
  `141423`); sim p99 of that gap 0.07. Cause unknown (balls or hand geometry vs the collision primitives).
- Stuck states = integrator windup: target = last target + 0.15·a, fingers blocked by the balls → target runs away.
  Real |target − q| ≈ 2× sim (index / middle median 0.21 vs 0.12 rad, p99 0.6–0.7 vs 0.35–0.46); `dagger_rel_p`
  stuck with the index 0.53 rad short all episode. hand_node interpolates each target over 50 ms, the sim steps it.
  A deploy-only clip to q ± δ does not work alone: δ = 0.5 clips 0.5–2% of sim steps but misses the stuck gaps
  (0.3–0.56); δ = 0.3 catches them but clips 11–36% of sim steps. → `--ramp` (targets interpolated over the control
  step) + `--windup 0.3` in training and in `baoding_server`.
- Ball size: `geometry(d)` keeps the wrist-side ball edge ≥ 40 mm from the hand origin (orbit center z = max(86,
  42 + d) mm; 40 mm unchanged), `start_pose(d)` opens the fingers until balls anywhere on the orbit overlap the hand
  ≤ 1.5 mm (40 mm: DEFAULT_POSE, 1.1 mm). Holdable with the hand still (nominal tilt, 64 episodes, 5 s, drop):
  40 mm 0.06, 50 mm 0.09, 52 mm 0.22, 54 mm 0.59, 55 mm 1.00 (flat start fingers, balls roll off the fingertips);
  60 mm overlaps even flat fingers. Ball stays inside the 88 px image up to 55 mm. Mass assumed 10 g × (d / 40)²
  (PLA shell): 46 mm 13.2 g, 52 mm 16.9 g; solid-sphere inertia as before.
- Warp physics is not bit-reproducible run to run (4e-5 after one step); the 40 mm default reproduces the old
  rollout statistics exactly (DR teacher, 64 episodes: pace 0.86, drop 0.14).
- `baoding_rerun`: raw 640×480 wrist frame (JPEG, ~1.6 ms) next to the policy input; only in Rerun / `.rrd`.

### Stage 1: direction × ball size (`pipeline8.sh`; teachers only, warm start from the DR teacher, 250M steps, 50 min each)
Same flags as the DR teacher (tilt 35° about nominal, DR, `--w_over 1`); evals: random = training distribution
(2 × 128 episodes, mean), nominal = 17° fingers downhill + DR (128 episodes). Thumb = share of steps the thumb
touches a ball (`--contacts`, CPU replay of 32 episodes). Repeats of the random eval differ by up to ±6 points.

| teacher | random: pace% / drop / thumb | nominal T = 4 s: pace% / drop / thumb · index · middle · ring · pinky | nominal T = 8 s: pace% / drop / thumb |
|---|---|---|---|
| baseline (DR teacher) | 84 / 0.14 / 0.06 | 100 / 0.00 / 0.00 · 0.56 · 0.94 · 0.93 · 0.59 | 100 / 0.00 / 0.01 |
| fwd 40 (control, +250M) | 84 / 0.15 / 0.07 | 100 / 0.00 / 0.00 · 0.61 · 0.94 · 0.91 · 0.61 | 100 / 0.00 / 0.04 |
| rev 40 | 82 / 0.15 / 0.01 | 100 / 0.00 / 0.00 · 0.87 · 0.52 · 0.91 · 0.81 | 100 / 0.00 / 0.00 |
| fwd 46 | 73 / 0.26 / 0.38 | 100 / 0.00 / 0.22 · 0.61 · 0.97 · 0.93 · 0.73 | 100 / 0.00 / 0.46 |
| rev 46 | 70 / 0.29 / 0.38 | 100 / 0.00 / 0.17 · 0.88 · 0.72 · 0.96 · 0.44 | 100 / 0.00 / 0.49 |
| fwd 52 | 54 / 0.46 / 0.82 | 98 / 0.02 / 0.76 · 0.57 · 0.92 · 0.97 · 0.74 | 99 / 0.01 / 0.80 |
| rev 52 | 52 / 0.45 / 0.74 | 99 / 0.01 / 0.54 · 0.76 · 0.61 · 0.84 · 0.32 | 99 / 0.00 / 0.68 |

- Conclusion 1: ball size engages the thumb, direction does not. Reversing at 40 mm hands the thumb-side uphill work
  to the index (0.56 → 0.87), the thumb stays at 0. The control shows the extra training alone changes nothing.
- Conclusion 2: the price of size is robustness off-nominal (random tilt: drop 0.14 → 0.26 at 46 mm, 0.46 at 52 mm);
  at the real nominal tilt all sizes keep ~100% pace with ≤ 0.02 drops. Reverse is equal or slightly worse
  everywhere (52 mm: thumb 0.54 vs 0.76 at T = 4 s, track error 9.8 vs 4.6 mm).
- Looks (video frames, T = 8 s): the DR teacher keeps the thumb folded back at the wrist; at 46 mm it lies along the
  thumb-side ball as a guide; at 52 mm it wraps around the side and over the balls.
- Decision: final = forward, 52 mm (most thumb, nominal performance intact), + route A.
