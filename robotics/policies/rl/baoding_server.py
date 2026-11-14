"""Policy server for the real HX5: a DAgger wrist-camera student behind ros2_ur scripts/deploy/hand_node.py.
Protocol: ros2_ur docs/deploy.md "Hand-only policies" (reset -> start pose; obs -> 20 absolute finger targets).
Tracking students: the target phase starts at the pair angle seen by the image-only ball head on the first frame
(or --theta0). --rel students (detected from the checkpoint: goal obs std ~0) get zero goal obs.
Each rollout is saved to out/rl/baoding_real/server/<start time>.npz at the next reset / disconnect / Ctrl-C.
Task options stored in the student (ball size -> orbit / start pose, reverse direction, anti-windup) are applied here.
--rerun: live Rerun view of each rollout, incl. the raw camera frame (baoding_rerun.py, which also replays the .npz
without raw frames); --save adds <start time>.rrd (the only copy of the raw frames).
Run (base conda): python -m robotics.policies.rl.baoding_server --student out/rl/baoding_real/dagger_rel/student.pkl --period 16"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import pickle
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import jax
import numpy as np

from platforms.paths import output_path
from robotics.policies.rl import baoding_env as E
from robotics.policies.rl.baoding_dagger import Student
from robotics.policies.rl.baoding_noisy_env import TH0, geometry, gravity_dir, start_pose
from robotics.policies.rl.baoding_real import undistort

sys.path.insert(0, "/home/lab/Documents/ros2_ur/scripts/deploy")
import policy_link  # noqa: E402

DT, ACTION_SCALE = 0.05, 0.15  # as BaodingNoisyEnv
LOG_DIR = output_path("rl/baoding_real/server")

p = argparse.ArgumentParser()
p.add_argument("--student", required=True)
p.add_argument("--period", type=float, default=16.0, help="commanded rotation period (s), trained on [2, 16]")
p.add_argument("--stack", type=int, default=3, help="stacked frames, as the checkpoint")
p.add_argument("--theta0", type=float, help="tracking students: fixed start phase (rad) instead of the image estimate")
p.add_argument("--rerun", action="store_true", help="show each rollout live in Rerun")
p.add_argument("--save", action="store_true", help="with --rerun: also write an .rrd per rollout")
a = p.parse_args()
if a.rerun:
    from robotics.policies.rl import baoding_rerun
name = Path(a.student).parent.name

d = pickle.load(open(a.student, "rb"))
mean, std = np.asarray(d["norm"][0]), np.asarray(d["norm"][1])
rel = bool(std[60:66].max() < 0.01)
model = Student(d.get("enc", "flat"))
forward = jax.jit(lambda pix, x: model.apply(d["params"], pix[None], x[None]))
lo, hi = E.build_model()[0].actuator_ctrlrange.T
task = d.get("task", {})
ball_d, sgn, windup = task.get("ball_d", 0.04), -1.0 if task.get("reverse") else 1.0, task.get("windup")
_, _, orbit_c, orbit_r = geometry(ball_d)
start = start_pose(ball_d)
g_meas = np.asarray(gravity_dir(TH0, 0.0))  # arm at the nominal baoding pose: 17 deg fingers downhill, roll 0
omega = 2 * np.pi / a.period


def infer(pix, vis_state):
    """-> action (20,), ball centers (2, 3) in the hand frame (m) from the image-only head."""
    act, ball = forward(pix, ((vis_state - mean) / std).astype(np.float32))
    return np.asarray(act[0], float), np.asarray(ball[0]).reshape(2, 3) / 10 + E.CENTER


def flush(s):
    log = s["log"]
    if log:
        path = LOG_DIR / f"{s['t0']}.npz"
        keys = ("frame", "vis_state", "action", "target", "balls", "theta", "stamps", "infer_ms")
        np.savez_compressed(path, student=name, period=a.period, **{k: np.array(v) for k, v in zip(keys, zip(*log))})
        print(f"{len(log)} steps -> {path}", flush=True)
    log.clear()


def step(o, s):
    """One obs -> absolute finger targets. s = rollout state: frames (newest first), theta, log."""
    t = time.perf_counter()
    img = undistort(o["wrist"])  # RGB stays RGB
    s["frames"] = ([img] + s["frames"][:-1]) if s["frames"] else [img] * a.stack
    pix = np.concatenate(s["frames"], -1)
    if s["theta"] is None:  # first obs: ball0 (white) sits at orbit angle theta, ball1 (yellow) opposite
        b = infer(pix, mean)[1]
        s["theta"] = a.theta0 if a.theta0 is not None else float(np.arctan2(b[0, 2] - b[1, 2], b[0, 1] - b[1, 1]))
        print(f"start phase {np.degrees(s['theta']):.0f} deg; image head (y, z mm from the orbit center): white "
              f"{(1e3 * (b[0, 1:] - orbit_c)).round()}, yellow {(1e3 * (b[1, 1:] - orbit_c)).round()}", flush=True)
    else:
        s["theta"] += sgn * omega * DT
    cs, sn = np.cos(s["theta"]), np.sin(s["theta"])
    tg = orbit_c + orbit_r * np.array([[cs, sn], [-cs, -sn]])  # = E.targets(theta, orbit_c, orbit_r)
    goal = np.zeros(6) if rel else np.array([*(10 * (tg - E.CENTER[1:])).ravel(), sn, cs])
    vis_state = np.concatenate([o["hand_q"], 0.1 * o["hand_qd"], o["hand_cmd"], goal, g_meas, [omega / np.pi]])
    act, balls = infer(pix, vis_state)
    target = o["hand_cmd"] + ACTION_SCALE * act
    if windup is not None:  # as the env: commanded target within +-windup of the measured joint angle
        target = np.clip(target, o["hand_q"] - windup, o["hand_q"] + windup)
    target = np.clip(target, lo, hi)
    s["log"].append((img, vis_state, act, target, balls, s["theta"], o["stamps"], (time.perf_counter() - t) * 1e3))
    return target


# JIT warm-up, and a check that the checkpoint matches this Student / --stack
act, _ = infer(np.zeros((64, 88, 3 * a.stack), np.uint8), mean)
assert act.shape == (20,) and np.all(np.isfinite(act))
print(f"{a.student}: {'rel (no orbit)' if rel else 'tracking'}, period {a.period} s, stack {a.stack}, task {task}", flush=True)

LOG_DIR.mkdir(parents=True, exist_ok=True)
state = {"frames": [], "theta": None, "log": [], "t0": None}  # t0 = rollout start time, names the log files
with Listener(policy_link.ADDRESS, authkey=policy_link.AUTHKEY) as listener:
    try:
        while True:
            print("waiting for hand_node ...", flush=True)
            with listener.accept() as conn:
                try:
                    while True:
                        cmd, o = policy_link.unpack(conn.recv_bytes())
                        if cmd == "reset":
                            flush(state)
                            state.update(frames=[], theta=None)
                            print(f"reset: place the {1e3 * ball_d:.0f} mm balls " + ("anywhere on the palm" if rel else
                                  "touching, around the palm center (white = ball0)"), flush=True)
                            conn.send_bytes(policy_link.pack("ok", {"start": start}))
                        elif cmd == "obs":
                            if not state["log"]:  # first step of a rollout
                                state["t0"] = t0 = time.strftime("%Y%m%d_%H%M%S")
                                if a.rerun:
                                    baoding_rerun.start(name, t0, LOG_DIR / f"{t0}.rrd" if a.save else None, raw=True)
                            conn.send_bytes(policy_link.pack("action", {"hand": step(o, state)}))
                            if a.rerun:  # after the reply
                                baoding_rerun.log(len(state["log"]) - 1, *state["log"][-1], raw=o["wrist"])
                except (EOFError, ConnectionResetError):
                    flush(state)
    except KeyboardInterrupt:
        flush(state)
