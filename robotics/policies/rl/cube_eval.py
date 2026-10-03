"""Run a trained cube policy in CPU MuJoCo, record video, report consecutive successes.
Example: MUJOCO_GL=egl python -m robotics.policies.rl.cube_eval --params out/rl/cube/params.pkl --episodes 5"""
from platforms.paths import output_path
import argparse
import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small MLP; leaves the GPU to training

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from robotics.policies.rl.cube_env import CUBE_START, DEFAULT_POSE, build_model, observe, quat_angle
from robotics.sim import Sim

p = argparse.ArgumentParser()
p.add_argument("--params", default=str(output_path('rl/cube/params.pkl')))
p.add_argument("--episodes", type=int, default=5)
p.add_argument("--video_episodes", type=int, default=2)
p.add_argument("--video", default=str(output_path('rl/cube.mp4')))
p.add_argument("--viewer", action="store_true")
p.add_argument("--goal", default="full", choices=["full", "z"])
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

OBS_SIZE = 81
net = ppo_networks.make_ppo_networks(OBS_SIZE, 20, preprocess_observations_fn=running_statistics.normalize,
                                     policy_hidden_layer_sizes=(512, 256, 128),
                                     value_hidden_layer_sizes=(512, 256, 128))
policy = jax.jit(ppo_networks.make_inference_fn(net)(pickle.load(open(a.params, "rb")), deterministic=True))
obs_fn = jax.jit(observe)

m = build_model(goal_marker=True)
cam = dict(lookat=np.array([0.1, 0.03, 0.36]), distance=0.5, azimuth=200, elevation=-35)
rng = np.random.default_rng(a.seed)
lo, hi = m.actuator_ctrlrange.T
n_frames, success_tol = 10, 0.4


def random_goal():
    if a.goal == "z":
        t = rng.uniform(-np.pi, np.pi)
        return np.array([np.cos(t / 2), 0, 0, np.sin(t / 2)])
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


results = []
for ep in range(a.episodes):
    s = Sim(m, viewer=a.viewer, video=a.video.replace(".mp4", f"_{ep}.mp4") if ep < a.video_episodes else None, cam=cam)
    s.reset(hand_q=np.clip(DEFAULT_POSE, lo, hi))
    s.d.qpos[20:27] = [*CUBE_START, 1, 0, 0, 0]
    goal = random_goal()
    s.d.mocap_quat[0] = goal
    mujoco.mj_forward(m, s.d)
    succ, dropped, key = 0, False, jax.random.PRNGKey(ep)
    for t in range(600):
        act, _ = policy(obs_fn(s.d.qpos, s.d.qvel, s.d.ctrl, goal), key)
        s.d.ctrl[:] = np.clip(s.d.ctrl + 0.15 * np.asarray(act), lo, hi)
        s.step(n_frames)
        pos, quat = s.d.qpos[20:23], s.d.qpos[23:27]
        if np.linalg.norm(pos - CUBE_START) > 0.1 or pos[2] < CUBE_START[2] - 0.04:
            dropped = True
            s.hold(0.5)
            break
        if float(quat_angle(jnp.asarray(goal), jnp.asarray(quat))) < success_tol:
            succ += 1
            goal = random_goal()
            s.d.mocap_quat[0] = goal
    s.close()
    results.append((succ, dropped, t + 1))
    print(f"episode {ep}: consecutive successes {succ}, dropped {dropped}, steps {t + 1}")

succ = np.array([r[0] for r in results])
print(f"mean successes {succ.mean():.2f} (max {succ.max()}), drop rate {np.mean([r[1] for r in results]):.2f}")
