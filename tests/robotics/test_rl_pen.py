"""Fast checks for the pen RL env (CPU MuJoCo only; MJX compile is too slow for a unit test)."""
import mujoco
import numpy as np

from robotics.policies.rl import pen_env as E
from robotics.sim import Sim


def test_pen_rests_across_fingers():
    s = Sim(E.build_model())
    s.reset(hand_q=E.DEFAULT_POSE)
    s.hold(1.0)
    assert not bool(E.dropped_fn(s.d.qpos))
    assert abs(float(E.pen_axis(s.d.qpos[23:27])[2])) < 0.1


def test_only_hand_pen_contacts_and_mjx_pairs():
    m = E.build_model()
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            assert m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX)


def test_obs_size():
    m = E.build_model()
    d = mujoco.MjData(m)
    assert E.observe(d.qpos, d.qvel, d.ctrl, 1.0).shape == (73,)
