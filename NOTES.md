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
| 7 | Screwdriver twirl with fingers only | scripted partial; RL next | partial: fixed-axis handle, index-only sweep gait, 467° in 12 cycles, one direction; thumb+index twirl failed (thumb CMC torque-limited) |
| 8 | Pen spinning | RL | todo |
| 9 | Baoding balls | RL | todo |
| 10 | Rubik's cube one-face turn | RL | todo |

## Decisions log
- 2026-09-23: replaced the mesh-collision combined model (`ur5e_hand.xml`) with MjSpec composition in `sim/scene.py`.
- 2026-09-23: own DLS IK instead of adding `mink` (6-DoF arm, one site target is enough).
- 2026-09-23: installed into base conda: pytest, mujoco-mjx 3.7.0, jax[cuda12] 0.11.2, brax 0.14.2 (RL uses MJX + brax PPO on GPU).
- 2026-09-23 consolidation: scene_base has `compiler angle="radian"` (spec-added joint ranges were silently degrees)
  and multiccd (flat cylinder ends; MJX accepts the flag); `arm_hand(base_height)` adds a pedestal; wrist servos kv 100 → 25
  (`scene.WRIST_KV`); ARM_HOME is now the wrist-up branch; Sim gained `cart_move` (lerp + slerp), `ik_high`
  (multi-seed, wrist above site, nearest to current ctrl) and `tip_ik`; postures CURL/PINCH/TRIPOD/KEY_PINCH/HOOK/POWER in sim/hand.py.
- Arm position servos sag ~1 cm under the hand's weight (no gravity compensation); tasks close the loop on object pose where it matters.

## Hand limits seen
- Fingers can't wrap thin handles alone (35.5 mm phalanges, 20 mm thick): objects must seat against the palm.
- Thumb opposes only in front of the palm; it can't face the index pad squarely (~55° off at yaw limit −1.571).
  Joint ranges match the vendored ROBOTIS model, so this is kept as-is.
- Adjacent straight fingers touch after ~0.05 rad inward MCP roll (index→middle); flexed 1.6 rad, ~0.15 rad.
  The vendored mesh model gives the same onsets within ~0.02 rad, so roll is mainly for spreading outward.
- Straight-thumb CMC barely moves the tip sideways; grip hold/slip threshold is ~0.01 rad of extra squeeze.
- RL cube: train `python -m rl.train_cube` (params → out/rl/cube/), eval `MUJOCO_GL=egl python -m rl.eval_cube` (videos out/rl/cube_*.mp4). Checkpoints are not in git.
