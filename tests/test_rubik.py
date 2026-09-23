"""Rubik's cube: scripted quarter turns on the fixed cube; RL env model layout (CPU, no MJX stepping)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import numpy as np

from sim import Sim
from tasks import rubik as R


def test_two_quarter_turns():
    s = R.add_detent(Sim(R.build()))
    s.reset(arm_q=s.ik("palm", R.palm_target() + [0, 0, 0.1], R.QUAT, q0=R.scene.ARM_HOME), hand_q=R.thumb(R.C0, R.UP))
    R.approach(s)
    assert R.quarter_turn(s, -90)
    assert R.quarter_turn(s, -180)


def test_rl_model_layout():
    from rl.rubik_env import CUBE_START, DEFAULT_POSE, build_model
    m = build_model()
    assert (m.nq, m.nv, m.nu) == (28, 27, 20)
    assert m.joint("rubik_top").qposadr[0] == 27 and m.joint("rubik_top").dofadr[0] == 26
    d = mujoco.MjData(m)
    d.qpos[:20], d.ctrl[:] = DEFAULT_POSE, DEFAULT_POSE
    d.qpos[20:28] = [*CUBE_START, 1, 0, 0, 0, 0]
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.linalg.norm(d.qpos[20:23] - CUBE_START) < 0.02  # cube rests on the palm
