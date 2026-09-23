"""Fast CPU checks for the cube RL env pieces (no MJX stepping)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import numpy as np

from rl.cube_env import CUBE_START, DEFAULT_POSE, build_model, observe, quat_angle


def test_model_layout_and_contacts():
    m = build_model(goal_marker=True)
    assert m.nq == 27 and m.nu == 20 and m.nmocap == 1
    d = mujoco.MjData(m)
    d.qpos[:20], d.ctrl[:] = DEFAULT_POSE, DEFAULT_POSE
    d.qpos[20:27] = [*CUBE_START, 1, 0, 0, 0]
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.linalg.norm(d.qpos[20:23] - CUBE_START) < 0.02  # cube rests on palm


def test_obs_and_angle():
    qpos, qvel = np.zeros(27), np.zeros(26)
    qpos[23] = 1
    assert observe(qpos, qvel, np.zeros(20), np.array([1.0, 0, 0, 0])).shape == (81,)
    q90 = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    assert abs(float(quat_angle(q90, np.array([1.0, 0, 0, 0]))) - np.pi / 2) < 1e-5
