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
| 1 | Multi-pose grasps (power / pinch / tripod / lateral / hook; top & side) | script, arm+hand | done: 5/5 lift 15 cm; payload upper bounds power ~6 kg (suspicious), pinch ~0.64 kg |
| 2 | Spray bottle trigger | script, arm+hand | done: 3/3 full pulls, bottle held |
| 3 | Hammer a nail | script, arm+hand | done: 2.2 cm in 8 strikes, 230 g hammer |
| 4 | Scissors open/close | script, hand-only | done: 5 cycles, 75–85% of range; paper cut faked by weld release |
| 5 | In-hand position alignment (slide/pivot a rod to a target) | script first, RL later | partial: slide −20 mm ±0.4, pivot +16° ±0.2, translate ±6 mm ±1 mm; one direction only, RL needed for general |
| 6 | Cube arbitrary reorientation | RL (hand-only, palm up) | todo |
| 7 | Screwdriver twirl with fingers only | RL (try scripted gait) | todo |
| 8 | Pen spinning | RL | todo |
| 9 | Baoding balls | RL | todo |
| 10 | Rubik's cube one-face turn | RL | todo |

## Decisions log
- 2026-09-23: replaced the mesh-collision combined model (`ur5e_hand.xml`) with MjSpec composition in `sim/scene.py`.
- 2026-09-23: own DLS IK instead of adding `mink` (6-DoF arm, one site target is enough).
- 2026-09-23: installed into base conda: pytest, mujoco-mjx 3.7.0, jax[cuda12] 0.11.2, brax 0.14.2 (RL uses MJX + brax PPO on GPU).
- Pending shared fixes (from hammer): ARM_HOME puts wrist links below the palm (use wrist-up IK branch); UR5e wrist servos over-damped (kv 100 → ~25).
- Pending shared fixes (from spray): Cartesian straight-line move (`cart_move`/`slerp` in tasks/spray.py) worth promoting into Sim.
- Hand limits seen: fingers can't wrap thin handles alone (35.5 mm phalanges, 20 mm thick) — objects must seat against palm; thumb only opposes in front of the palm.
- Pending shared fix (from scissors): scene_base.xml lacks `<compiler angle="radian">`, so joint ranges added via MjSpec are read as degrees (scissors uses np.degrees workaround).
- Hand limits seen: index MCP roll toward middle hits middle knuckle box; straight-thumb CMC barely moves the tip sideways.
- Pending shared fixes (from grasps): arm base pedestal (~0.25 m) for table at 0.4 m; multiccd for flat-bottom cylinders; promote postures + cart_move/ik_high into sim.
- Hand limits seen (align): thumb can't face index pad squarely (~55° off at yaw limit); adjacent knuckle boxes overlap ~7 mm at small roll → roll range unusable; grip hold/slip threshold ~0.01 rad of squeeze.
