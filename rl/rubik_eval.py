"""Run a trained Rubik policy in CPU MuJoCo, record video, report quarter turns per episode.
Example: MUJOCO_GL=egl python -m rl.rubik_eval --params out/rl/rubik/params.pkl --episodes 5"""
import argparse
import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small MLP; leaves the GPU to training

import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from rl.rubik_env import CUBE_START, DEFAULT_POSE, QUARTER, build_model, observe
from sim import Sim
from tasks.rubik import add_detent

p = argparse.ArgumentParser()
p.add_argument("--params", default="out/rl/rubik/params.pkl")
p.add_argument("--episodes", type=int, default=5)
p.add_argument("--video_episodes", type=int, default=2)
p.add_argument("--video", default="out/rl/rubik.mp4")
a = p.parse_args()

net = ppo_networks.make_ppo_networks(76, 20, preprocess_observations_fn=running_statistics.normalize,
                                     policy_hidden_layer_sizes=(512, 256, 128),
                                     value_hidden_layer_sizes=(512, 256, 128))
policy = jax.jit(ppo_networks.make_inference_fn(net)(pickle.load(open(a.params, "rb")), deterministic=True))
obs_fn = jax.jit(observe)

m = build_model(stickers=True)
cam = dict(lookat=np.array([0.08, 0.0, 0.36]), distance=0.4, azimuth=200, elevation=-35)
lo, hi = m.actuator_ctrlrange.T
tol = np.radians(10)

results = []
for ep in range(a.episodes):
    s = add_detent(Sim(m, video=a.video.replace(".mp4", f"_{ep}.mp4") if ep < a.video_episodes else None, cam=cam))
    s.reset(hand_q=np.clip(DEFAULT_POSE, lo, hi))
    s.d.qpos[20:28] = [*CUBE_START, 1, 0, 0, 0, 0]
    sign = 1.0 if ep % 2 == 0 else -1.0
    target, turns, dropped, key = sign * QUARTER, 0, False, jax.random.PRNGKey(ep)
    for t in range(600):
        act, _ = policy(obs_fn(s.d.qpos, s.d.qvel, s.d.ctrl, target), key)
        s.d.ctrl[:] = np.clip(s.d.ctrl + 0.15 * np.asarray(act), lo, hi)
        s.step(10)
        if np.linalg.norm(s.d.qpos[20:23] - CUBE_START) > 0.1 or s.d.qpos[22] < CUBE_START[2] - 0.04:
            dropped = True
            s.hold(0.5)
            break
        if abs(s.d.qpos[27] - target) < tol and abs(s.d.qvel[26]) < 1.0:
            turns += 1
            target += sign * QUARTER
    s.close()
    results.append((turns, dropped))
    print(f"episode {ep}: quarter turns {turns}, dropped {dropped}, steps {t + 1}, layer {np.degrees(s.d.qpos[27]):.0f} deg")

print(f"mean quarter turns {np.mean([r[0] for r in results]):.2f}, drop rate {np.mean([r[1] for r in results]):.2f}")
