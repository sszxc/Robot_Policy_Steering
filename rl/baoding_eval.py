"""Run a trained Baoding policy in CPU MuJoCo, record video, report pair rotation and drops.
Example: MUJOCO_GL=egl python -m rl.baoding_eval --params out/rl/baoding/params.pkl"""
import argparse
import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small MLP; leaves the GPU to training

import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from rl import baoding_env as E
from sim import Sim

p = argparse.ArgumentParser()
p.add_argument("--params", default="out/rl/baoding/best_params.pkl")
p.add_argument("--period", type=float, default=16.0)
p.add_argument("--episodes", type=int, default=5)
p.add_argument("--video", default="out/rl/baoding.mp4")
p.add_argument("--viewer", action="store_true")
a = p.parse_args()

net = ppo_networks.make_ppo_networks(78, 20, preprocess_observations_fn=running_statistics.normalize,
                                     policy_hidden_layer_sizes=(512, 256, 128),
                                     value_hidden_layer_sizes=(512, 256, 128))
policy = jax.jit(ppo_networks.make_inference_fn(net)(pickle.load(open(a.params, "rb")), deterministic=True))
obs_fn = jax.jit(E.observe)

m, R, o = E.build_model()
lo, hi = m.actuator_ctrlrange.T
omega = 2 * np.pi / a.period
cam = dict(lookat=o + R @ E.CENTER, distance=0.32, azimuth=180, elevation=-60)
turns, rng = [], np.random.default_rng(0)
for ep in range(a.episodes):
    s = Sim(m, viewer=a.viewer, video=a.video if ep == 0 else None, cam=cam)
    s.reset(hand_q=np.clip(E.DEFAULT_POSE, lo, hi))
    if ep > 0:  # same 3 mm in-plane ball noise as the MJX env reset
        for adr in (20, 27):
            s.d.qpos[adr:adr + 3] += R @ (np.array([0, 1, 1]) * rng.normal(0, 0.003, 3))
    b = np.asarray(E.balls_hand(s.d.qpos, R, o))
    pair, turn, theta, dropped, key = float(E.pair_angle(b)), 0.0, 0.0, False, jax.random.PRNGKey(ep)
    trace = [b]
    for t in range(400):
        act, _ = policy(obs_fn(s.d.qpos, s.d.qvel, s.d.ctrl, theta, R, o), key)
        s.d.ctrl[:] = np.clip(s.d.ctrl + 0.15 * np.asarray(act), lo, hi)
        s.step(10)
        theta += omega * 0.05
        b = np.asarray(E.balls_hand(s.d.qpos, R, o))
        new = float(E.pair_angle(b))
        turn += (new - pair + np.pi) % (2 * np.pi) - np.pi
        pair = new
        trace.append(b)
        if bool(E.dropped_fn(b)):
            dropped = True
            s.hold(0.5)
            break
    s.close()
    turns.append(turn)
    tr = np.array(trace)[:, :, 1:]  # (T, 2, yz)
    path = 1000 * np.linalg.norm(np.diff(tr, axis=0), axis=-1).sum(0)
    mid = 1000 * tr.mean(1)
    print(f"episode {ep}: pair rotation {np.degrees(turn):.0f} deg in {(t + 1) * 0.05:.1f} s "
          f"(targets {np.degrees(theta):.0f} deg), half-swaps {int(abs(turn) // np.pi)}, dropped {dropped}, "
          f"ball paths {path.round()} mm, midpoint std {mid.std(0).round(1)} mm")
print(f"mean rotation {np.degrees(np.mean(turns)):.0f} deg")
