"""Fast checks for the screwdriver RL env (CPU MuJoCo only; no MJX jit)."""
import mujoco
import numpy as np
from mujoco import mjx

from robotics.policies.rl.screwdriver_env import DEFAULT_POSE, build_model, observe


def test_model_is_mjx_compatible():
    mjx.put_model(build_model())


def test_default_pose_touches_handle():
    m = build_model()
    d = mujoco.MjData(m)
    d.qpos[:20] = d.ctrl[:] = DEFAULT_POSE
    for _ in range(200):
        mujoco.mj_step(m, d)
    bodies = {m.body(m.geom_bodyid[c.geom1]).name for c in d.contact[: d.ncon]}
    assert {"finger_r_link4", "finger_r_link8"} <= bodies
    assert min((c.dist for c in d.contact[: d.ncon]), default=0) > -0.003


def test_obs_size():
    m = build_model()
    d = mujoco.MjData(m)
    assert np.asarray(observe(d.qpos, d.qvel, d.ctrl, 1.0)).shape == (64,)
