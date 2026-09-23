"""Run a trained screwdriver policy in CPU MuJoCo (both directions), record video, report speed and contacts.
Example: MUJOCO_GL=egl python -m rl.eval_screwdriver --params out/rl/screwdriver/params.pkl"""
import argparse
import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small MLP; leaves the GPU to training

import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from rl.screwdriver_env import AXIS, DEFAULT_POSE, build_model, observe
from sim import Sim

p = argparse.ArgumentParser()
p.add_argument("--params", default="out/rl/screwdriver/params.pkl")
p.add_argument("--seconds", type=float, default=10.0, help="per direction")
p.add_argument("--video", default="out/rl/screwdriver.mp4")
p.add_argument("--directions", default="1,-1")
a = p.parse_args()

net = ppo_networks.make_ppo_networks(64, 20, preprocess_observations_fn=running_statistics.normalize,
                                     policy_hidden_layer_sizes=(512, 256, 128),
                                     value_hidden_layer_sizes=(512, 256, 128))
policy = jax.jit(ppo_networks.make_inference_fn(net)(pickle.load(open(a.params, "rb")), deterministic=True))
obs_fn = jax.jit(observe)

m = build_model()
lo, hi = m.actuator_ctrlrange.T
cam = dict(lookat=np.array([AXIS[0], AXIS[1], 0.2]), distance=0.42, azimuth=145, elevation=-22)
s = Sim(m, video=a.video, cam=cam)
key = jax.random.PRNGKey(0)
for direction in [float(d) for d in a.directions.split(",")]:
    s.reset(hand_q=np.clip(DEFAULT_POSE, lo, hi))
    a0, touch, n = s.d.qpos[20], {}, int(a.seconds / 0.05)
    for t in range(n):
        act, _ = policy(obs_fn(s.d.qpos, s.d.qvel, s.d.ctrl, direction), key)
        s.d.ctrl[:] = np.clip(s.d.ctrl + 0.15 * np.asarray(act), lo, hi)
        s.step(10)
        for b in s.hand_contacts("screwdriver"):
            touch[b] = touch.get(b, 0) + 1
    speed = np.degrees(direction * (s.d.qpos[20] - a0)) / a.seconds
    top = sorted(touch.items(), key=lambda kv: -kv[1])[:6]
    print(f"direction {direction:+.0f}: {speed:6.1f} deg/s in commanded direction, total "
          f"{np.degrees(direction * (s.d.qpos[20] - a0)):.0f} deg; contact fraction "
          + ", ".join(f"{b} {c / n:.2f}" for b, c in top))
s.close()
