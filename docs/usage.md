# Usage

Install the project editable (`python -m pip install --no-build-isolation --no-deps -e .`). Commands can then run from any directory; default outputs are resolved from the repository root or `THESIS_OUTPUT_ROOT`. Explicit relative output/checkpoint paths remain relative to the current directory. Headless video needs `MUJOCO_GL=egl`. `robotics.sim.view` and `--viewer` need a display, so leave that variable unset.

## View a scene

```
python -m robotics.sim.view [hand|arm|lab]
```

Default is `arm` (UR5e + hand). `hand` is the hand alone. `lab` is the real workcell (`scene.lab_scene()`, from ros2_ur):
calibrated UR5 on its stand, HX5 on the -90° mount, table, wall and camera rig, in the ROS `world` frame, starting at
the ros2_ur palm home (`scene.LAB_HOME`). Sites `tool0`, `palm_center` and `table_surface_center` match the ROS frames.

## Scripted tasks

```
python -m robotics.policies.scripted.<name> [--viewer] [--video [path]] [--log path.npz]
```

`--viewer` opens a MuJoCo window. `--video` records `out/scripted/<name>/<name>.mp4`; `--video path.mp4` records that file. `--log` writes the trajectory to an npz.

| Command | What it does | Extra arguments |
|---|---|---|
| `robotics.policies.scripted.grasps` | Five grasps: pick, lift 15 cm, hold 2 s, put back | `--grasp power\|pinch\|tripod\|lateral\|hook` (omit to run all five). `--payload` bisects the heaviest mass (kg) still held, then exits; with no `--grasp`, only `power` and `pinch` |
| `robotics.policies.scripted.spray` | Grasp a spray bottle and pull the trigger 3 times with the index finger | |
| `robotics.policies.scripted.hammer` | Grasp a hammer and drive a nail | |
| `robotics.policies.scripted.scissors` | Open and close scissors with the fingers | |
| `robotics.policies.scripted.align` | In-hand rod alignment: slide, pivot, translate | |
| `robotics.policies.scripted.screwdriver` | Finger-twirl a screwdriver (clockwise only) | |
| `robotics.policies.scripted.baoding` | Orbit two balls with a fixed finger gait | `--search`: CEM search, then exit (80 iters, population 256, elite 24, 16 workers, seed 0; not configurable). Best params go to `out/scripted/baoding/best.npy`. Ignores `--viewer` / `--video` / `--log` |
| `robotics.policies.scripted.rubik` | Cube fixed on the table; the thumb turns the top layer | `--turns N` (default 4), 90° each |

## Reinforcement learning

Training runs on GPU (MJX + brax PPO). Logs append to `<out>/log.txt`. Each eval overwrites `<out>/params.pkl` (latest) and also writes `<out>/params_<step>.pkl`. Training does not create `best_params.pkl`; some evals default to that name, so copy a checkpoint there yourself. Weights are not in git.

On a shared GPU, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3` so JAX does not take all the memory.

Shared arguments: `--steps` env steps, `--envs` parallel envs, `--evals` evals during training, `--out` output directory, `--restore path.pkl` warm start (default: none).

| Command | Defaults | Extra arguments |
|---|---|---|
| `python -m robotics.policies.rl.cube_train` | `--steps 150000000 --envs 4096 --evals 30 --out out/rl/cube` | `--goal full\|z` (default `full`: any orientation; `z`: yaw about the vertical axis only) |
| `python -m robotics.policies.rl.screwdriver_train` | `--steps 150000000 --envs 4096 --evals 30 --out out/rl/screwdriver` | `--direction 0`: `+1` / `-1` fixed, `0` random each episode |
| `python -m robotics.policies.rl.pen_train` | `--steps 150000000 --envs 4096 --evals 30 --out out/rl/pen` | `--condim 3` contact dimension, `--w_rate 2.0` spin-rate reward weight, `--drop_penalty 10.0` |
| `python -m robotics.policies.rl.baoding_train` | `--steps 80000000 --envs 2048 --evals 25 --out out/rl/baoding` | `--period 8` seconds per target revolution, `--w_turn 10` rotation-progress reward weight |
| `python -m robotics.policies.rl.rubik_train` | `--steps 60000000 --envs 4096 --evals 30 --out out/rl/rubik` | |
| `python -m robotics.policies.rl.baoding_noisy_train` | `--steps 50000000 --envs 2048 --evals 25 --out out/rl/baoding_noisy/teacher` | Noisy env, pace-capped reward, critic sees privileged state. `--vision`: top-down 64 px RGB actor (defaults `--envs 1024` (512 fits a 16 GB laptop GPU), `--out out/rl/baoding_noisy/vision_rl`); `--period 16`, `--w_turn 50` |
| `python -m robotics.policies.rl.baoding_dagger` | `--teacher out/rl/baoding_noisy/teacher/best_params.pkl --iters 10 --envs 192 --grad_steps 3000 --out out/rl/baoding_noisy/dagger` | Distills the teacher into an RGB student (`student.pkl`); `--w_aux 0.1` ball-position loss. Keeps all frames in RAM (~1 GB per iteration) |
| v2: speed command + MuJoCo Warp | `baoding_noisy_train --periods 2 16 --warp` (teacher, 80M steps ≈ 17 min, `--out` default `out/rl/baoding_cmd/teacher`); add `--vision --envs 512 --w_over 1` for vision PPO (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`; without `--w_over` it learned to over-spin); `baoding_dagger --periods 2 16 --warp --teacher out/rl/baoding_cmd/teacher/best_params.pkl` | `--periods LO HI`: per-episode period (s/rev), log-uniform, ω/π appended to every obs. `--warp`: MJX-Warp physics + mesh/light rendering (needs `mujoco-warp==3.7.0`, `warp-lang==1.12.0`). Don't start a second GPU job next to a near-full one: a failed Warp allocation shows up as `CUDA_ERROR_ILLEGAL_ADDRESS` |

### CPU eval (video)

```
MUJOCO_GL=egl python -m robotics.policies.rl.<name> [--params path.pkl] ...
```

`--viewer` opens a MuJoCo window on the five CPU evals below. It needs a display, so leave `MUJOCO_GL` unset. Video recording is unchanged.

| Command | Arguments (defaults included) | Video |
|---|---|---|
| `robotics.policies.rl.cube_eval` | `--params out/rl/cube/params.pkl`, `--episodes 5`, `--video_episodes 2`, `--video out/rl/cube/cube.mp4`, `--viewer`, `--goal full\|z` (default `full`), `--seed 0` | First `--video_episodes` are `out/rl/cube/cube_0.mp4`, `cube_1.mp4` (`.mp4` replaced by `_<episode>.mp4`) |
| `robotics.policies.rl.screwdriver_eval` | `--params out/rl/screwdriver/params.pkl`, `--seconds 10` (per direction), `--video out/rl/screwdriver/screwdriver.mp4`, `--viewer`, `--directions 1,-1` (comma-separated, run in order) | Every direction goes into the same file |
| `robotics.policies.rl.pen_eval` | `--params out/rl/pen/best_params.pkl`, `--episodes 10`, `--video out/rl/pen/pen.mp4`, `--viewer`, `--condim 3` (must match training) | Episodes 0 and 1 (CCW, CW) are `out/rl/pen/pen_ccw.mp4`, `pen_cw.mp4` |
| `robotics.policies.rl.pen_check` | `--params out/rl/pen/best_params.pkl`, `--dir 1` (`+1` CCW, `-1` CW), `--seconds 20` | Prints whether the policy is exploiting the reward |
| `robotics.policies.rl.baoding_eval` | `--params out/rl/baoding/best_params.pkl`, `--period 16` (seconds per target revolution; training default is 8), `--episodes 5`, `--video out/rl/baoding/baoding.mp4`, `--viewer` | Only episode 0 |
| `robotics.policies.rl.rubik_eval` | `--params out/rl/rubik/params.pkl`, `--episodes 5`, `--video_episodes 2`, `--video out/rl/rubik/rubik.mp4`, `--viewer` | Same as cube: `out/rl/rubik/rubik_0.mp4`, `rubik_1.mp4` |

### GPU batched eval

`--params` takes one or more pkl paths, space-separated.

| Command | Arguments (defaults included) |
|---|---|
| `python -m robotics.policies.rl.pen_mjx_eval` | `--params out/rl/pen/best_params.pkl`, `--episodes 64` |
| `python -m robotics.policies.rl.baoding_mjx_eval` | `--params out/rl/baoding/best_params.pkl`, `--period 16`, `--episodes 64` |
| `python -m robotics.policies.rl.baoding_check` | `--params out/rl/baoding/best_params.pkl`, `--period 16`, `--episodes 64`: normal vs ball obs frozen at reset vs a mid-episode kick |
| `MUJOCO_GL=egl python -m robotics.policies.rl.baoding_noisy_eval` | `--teacher` / `--vision_rl` / `--student` (any subset), `--episodes 64`, `--clean` (noise off), `--video` → `out/rl/baoding_noisy/<name>.mp4` (MuJoCo view + 64 px policy view). Vision policies also run with pixels frozen at reset. `--cmd` (v2 checkpoints): speed grid `--grid 2 3 4 8 16` (s/rev), command switch 8↔3 s at 10 s, `--video` → `out/rl/baoding_cmd/<name>_sweep.mp4` (16→8→4→2 s/rev, 5 s each); `--episodes` must match the batch (Warp render context size) |

### Smoke

```
python -m robotics.policies.rl.baoding_smoke
python -m robotics.policies.rl.rubik_smoke [--envs 512] [--steps 50]
```

## Tests

```
python -m pytest tests -q
```

## Teleop (Leap Motion -> floating hand)

Needs a display and a Leap device. Keys are read globally (pynput), so the viewer does not need focus.
`teleop` also opens a top-down Leap skeleton window (`leap_view.py`, from the leapc-python-bindings visualiser); closing either window quits. Points are coloured by height above the wrist (red near, blue far); `f` toggles skeleton/dots.

```
python -m robotics.teleop.teleop [--scale 1.0] [--alpha 0.5] [--wrist leap|optitrack]
python -m robotics.teleop.replay out/teleop/<file>.npz [--speed 1.0] [--loop] [--fix_wrist]
```

| Command | Keys | Notes |
|---|---|---|
| `robotics.teleop.teleop` | `t` freeze wrist position / resume (offset recomputed, no jump), `r` start/stop recording, `q` quit | `--scale` wrist translation scale, `--alpha` finger EMA weight (1 = no filter). Rotation always follows the hand. Recordings go to `out/teleop/<time>.npz` (raw Leap keypoints `kp` (N,26,3) + time `t`). `--wrist optitrack` is a stub |
| `robotics.teleop.replay` | `space` pause, `q` quit | Re-runs the current `robotics/teleop/mapping.py` on the raw keypoints. `--fix_wrist` keeps the wrist at home |

Joint mapping gains/offsets: `GAIN` / `OFFSET` in `robotics/teleop/mapping.py`.

## HX5 real hand (Leap teleop)

The vendor stack runs in the `robotis_hand` container (ROS 2 Jazzy, zenoh); the host talks to it over localhost UDP.
`robotics/hardware/hx5/bridge.py` runs in the container (thesis is mounted read-only at `/thesis`) and streams
targets to `hand_r_controller`: clamps to vendor limits, rate-limits (`--max_vel`, 2 rad/s) from the measured pose,
and stops publishing after `--timeout` (0.2 s) without packets, so the hand holds its last goal.

```bash
cd ~/Documents/robotis_hand/docker && ./container.sh enter      # one terminal each:
zenohd
ros2 launch robotis_hand_bringup hx5_d20_right.launch.py           # add use_mock_hardware:=true for a dry run
python3 /thesis/robotics/hardware/hx5/bridge.py                    # interactive shell already sets RMW_IMPLEMENTATION
# host (base env)
python -m robotics.hardware.hx5.sweep [--joints 1 6] [--amp 0.3]  # sim + real, one joint at a time: compare signs
python -m robotics.teleop.teleop --real                            # e = start/stop sending to the real hand
```

The thesis mount is one extra line in `robotis_hand/docker/docker-compose.yml` (`../../thesis:/thesis:ro`);
after changing the compose file recreate the container: `./container.sh stop` (answer y), then `./container.sh start`.

## UR5 hardware examples

Use the laptop environment with `ur_rtde`. Importing these modules does not
connect to the robot; executing their entrypoints connects to the selected IP.
`--help` works without the device dependency.

```bash
python -m robotics.hardware.ur5.ur5_read --ip 192.168.1.132
python -m robotics.hardware.ur5.ur5_move --ip 192.168.1.132
python -m robotics.hardware.ur5.ur5_move_servo --ip 192.168.1.132
```

`ur5_read` prints joint/TCP state. `ur5_move` executes the existing TCP circle
and returns to its starting pose. `ur5_move_servo` executes the existing joint
sine cycle using the UR5 CB3 example's 125 Hz period. Choose the device address
with `--ip` or `THESIS_UR5_IP` (default `192.168.1.132`).

## SmolVLA / LIBERO

Select the SmolVLA/LIBERO environment with editable LeRobot installed, then:

```bash
TASK_IDS='[0]' EPISODES=1 bash experiments/libero/run_libero10_smolvla.sh
```

The launcher retains LIBERO-10 relative actions and seed 1000 by default.
`POLICY`, `TASK_IDS`, `EPISODES`, `SEED`, `EPISODE_LENGTH`, and `OUTPUT_DIR` can
override the run inputs. Default results receive a new directory under `out/libero/smolvla/vanilla/`;
an explicitly selected directory must also be new. Each run records provenance.
For HPC submission use [the quick reference](ASU_Sol_HPC_quick.md).
