"""Sanity tests for the models and Sim helpers. Run: python -m pytest tests -q"""
import mujoco
import numpy as np

from sim import Sim, hand as H, scene


def _self_contacts(s):
    names = [(s.m.body(s.m.geom_bodyid[c.geom1]).name, s.m.body(s.m.geom_bodyid[c.geom2]).name)
             for c in s.d.contact[: s.d.ncon]]
    return [n for n in names if all(x.startswith(("finger_r_", "hand_base")) for x in n)]


def test_models_compile():
    assert scene.hand_only().compile().nu == 20
    m = scene.arm_hand().compile()
    assert (m.nq, m.nu) == (26, 26)


def test_hand_collisions_are_primitives():
    m = scene.arm_hand().compile()
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            assert m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH, m.geom(g).name


def test_open_hand_rests_without_self_contact():
    s = Sim(scene.hand_only().compile())
    s.reset(hand_q=H.OPEN)
    s.hold(1.0)
    assert not _self_contacts(s)
    assert np.abs(s.d.qvel).max() < 1e-3


def test_hand_tracks_targets():
    s = Sim(scene.hand_only().compile())
    s.reset(hand_q=H.OPEN)
    q = H.pose(thumb=(0.5, -0.5, 0.5, 0.5), index=(0, 1.0, 0.8, 0.5), middle=(0, 0.5, 0.5, 0.5))
    s.move_hand(q, 0.5)
    s.hold(0.5)
    assert np.abs(s.hand_q() - q).max() < 0.05


def test_arm_ik_and_tracking():
    s = Sim(scene.arm_hand().compile())
    s.reset(arm_q=scene.ARM_HOME, hand_q=H.OPEN)
    s.hold(0.5)
    assert np.linalg.norm(s.site_pos("palm") - [0.5, 0, 0.6]) < 0.01
    quat = scene.hand_quat((0, 0, -1), (1, 0, 0))
    s.move_site("palm", (0.45, 0.1, 0.5), quat, duration=1.0)
    s.hold(0.5)
    assert s.ik_err < 1e-3
    assert np.linalg.norm(s.site_pos("palm") - [0.45, 0.1, 0.5]) < 0.01
