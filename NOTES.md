# HX5-D20 + UR5e MuJoCo tasks — notes

Goal: get a feel for the ROBOTIS HX5-D20 right hand in sim (UR5e arm). Privileged state only (no cameras),
no domain randomization. RL comes later, so collisions are primitives and hand-only scenes exist.

## Layout
- `mujoco/hx5_right.xml` — hand with primitive collisions (source of truth; vendored `robotis_hand/` untouched).
- `mujoco/scene_base.xml`, `mujoco/objects/*.xml` — world and task objects.
- `sim/scene.py` — `hand_only(pos, quat)`, `arm_hand()`, `add_table`, `add_object`, `hand_quat(palm_dir, finger_dir)`.
- `sim/runner.py` — `Sim`: `reset`, `move(arm_q, hand_q, duration)`, `ik`/`ik_high`/`move_site`, `cart_move` (straight line), `tip_ik` (fingertips), `hold`, video/viewer/log.
- `sim/hand.py` — 20-dim hand target layout and named postures.
- `tasks/*.py` — one script per task; `python -m tasks.<name> [--viewer] [--video out/x.mp4]`.
- `view.py` — interactive viewer for base scenes. Tests: `python -m pytest tests -q`.

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
| 6 | Cube arbitrary reorientation | RL (hand-only, palm up) | done: MJX+PPO 150M steps (~2.5 h); eval 4.8 consecutive successes (0.4 rad tol) per 30 s, 0 drops |
| 7 | Screwdriver twirl with fingers only | scripted + RL | RL done: 228–268 deg/s CPU (251–280 MJX), both directions, thumb/index/middle gaiting; scripted: index-only 32 deg/s one direction |
| 8 | Pen spinning (yaw about palm normal) | RL | done: ~5 rad/s both directions (~16 turns / 20 s); MJX 0/64 drops, CPU 1/10; exploit checks passed |
| 9 | Baoding balls | RL (scripted failed) | done: ~22 half-swaps per 20 s (CPU, 1/10 drops; MJX 65.8 rad, 5% drops); fast flicking gait, ignores 16 s target pace |
| 10 | Rubik's cube one-face turn | scripted (fixture) done; in-hand RL not solved | done (fixture): 4 consecutive quarter turns, 1 thumb stroke each; in-hand RL 0.25 turns/ep, see report |

## Decisions log
- 2026-09-23: replaced the mesh-collision combined model (`ur5e_hand.xml`) with MjSpec composition in `sim/scene.py`.
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
- RL cube: train `python -m rl.train_cube` (params → out/rl/cube/), eval `MUJOCO_GL=egl python -m rl.eval_cube` (videos out/rl/cube_*.mp4). Checkpoints are not in git.

## Task #10 report — Rubik's cube one-face turn (2026-09-23)
**Scripted, cube in a table fixture (`tasks/rubik.py`): works.** Arm brings the palm down over a fixed cube; only the thumb
sweeps the top layer (CMC −1.4 → 1.0, retract outside), closed-loop on the hinge angle, plus a 4-fold detent (0.01 Nm peak).
4 consecutive quarter turns, one stroke each: −89.4 / −178.4 / −267.3 / −357.0°. Robustness: 24/27 cube offsets within ±6 mm pass
(6/27 without the detent, so the snap-to-quarter behaviour matters). Scripted in-hand (palm up) failed: one turn at most, very sensitive.

**RL, cube free in the palm-up hand (`rl/rubik_env.py`, `rubik_train.py`, `rubik_eval.py`): not solved.**
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
- RL screwdriver: `python -m rl.train_screwdriver` (152M steps, 3.3 h), eval `python -m rl.eval_screwdriver` → out/rl/screwdriver.mp4. RL model uses a capsule handle (MJX lacks cylinder–box). Caveats: fingertips penetrate up to 4.8 mm (soft contact, 5 solver iters); thumb/index/middle run at the 0.387 Nm limit.
- RL baoding: two-stage PPO (13.8M steps holding at 8 s/rev, then 51M at 16 s/rev with progress weight 50); best checkpoint 38.3M → out/rl/baoding/best_params.pkl, eval `python -m rl.baoding_eval` → out/rl/baoding.mp4. Scripted CEM gait failed (144° in 20 s): curled fingertips only reach the finger-side half of the palm, so nothing pushes balls toward the wrist.
- RL pen: pen rests on the proximal phalanges of slightly curled fingers (flat on the palm it can't be pushed sideways → first 110M-step run learned to hold still). Reward 2·yaw rate (clip ±3 rad/s), small offset/tilt terms, drop −10; NaN-safe obs. 204M steps (~2.1 h), best 183.5M → out/rl/pen/best_params.pkl; eval `python -m rl.pen_eval`, exploit checks `python -m rl.pen_check` → out/rl/pen.mp4. Index/middle/ring drive CCW, ring/pinky CW; thumb unused; pen also rolls about its own axis while yawing.
