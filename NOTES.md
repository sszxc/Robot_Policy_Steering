# HX5-D20 + UR5e MuJoCo tasks — notes

Goal: get a feel for the ROBOTIS HX5-D20 right hand in sim (UR5e arm). Privileged state only (no cameras),
no domain randomization. RL comes later, so collisions are primitives and hand-only scenes exist.

## Layout
- `mujoco/hx5_right.xml` — hand with primitive collisions (source of truth; vendored `robotis_hand/` untouched).
- `mujoco/scene_base.xml`, `mujoco/objects/*.xml` — world and task objects.
- `sim/scene.py` — `hand_only(pos, quat)`, `arm_hand()`, `add_table`, `add_object`, `hand_quat(palm_dir, finger_dir)`.
- `sim/runner.py` — `Sim`: `reset`, `move(arm_q, hand_q, duration)`, `ik`/`move_site`, `hold`, video/viewer/log.
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
| 1 | Multi-pose grasps (power / pinch / tripod / lateral / hook; top & side) | script, arm+hand | todo |
| 2 | Spray bottle trigger | script, arm+hand | todo |
| 3 | Hammer a nail | script, arm+hand | todo |
| 4 | Scissors open/close | script, hand-only | todo |
| 5 | In-hand position alignment (slide/pivot a rod to a target) | script first, RL later | todo |
| 6 | Cube arbitrary reorientation | RL (hand-only, palm up) | todo |
| 7 | Screwdriver twirl with fingers only | RL (try scripted gait) | todo |
| 8 | Pen spinning | RL | todo |
| 9 | Baoding balls | RL | todo |
| 10 | Rubik's cube one-face turn | RL | todo |

## Decisions log
- 2026-09-23: replaced the mesh-collision combined model (`ur5e_hand.xml`) with MjSpec composition in `sim/scene.py`.
- 2026-09-23: own DLS IK instead of adding `mink` (6-DoF arm, one site target is enough).
