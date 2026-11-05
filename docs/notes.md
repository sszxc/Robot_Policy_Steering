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
