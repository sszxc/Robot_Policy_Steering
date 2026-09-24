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


def test_adjacent_fingers_touch_only_after_inward_roll():
    # Straight index rolled toward middle: tips meet near 0.05 rad (vendored mesh model agrees within ~0.02).
    s = Sim(scene.hand_only().compile())
    for roll, expect in ((0.02, False), (0.1, True), (-0.61, False)):
        q = H.OPEN.copy()
        q[H.IDX["index"].start] = roll
        s.reset(hand_q=q)
        assert bool(_self_contacts(s)) == expect, roll


def test_spec_joint_ranges_are_radians():
    spec = scene.hand_only()
    b = spec.worldbody.add_body(name="probe", pos=[1, 0, 0.1])
    b.add_joint(name="probe", range=[0, 0.8])
    b.add_geom(size=[0.01, 0, 0])
    assert abs(spec.compile().joint("probe").range[1] - 0.8) < 1e-9


def test_ik_high_keeps_wrist_up_and_cart_move_is_straight():
    s = Sim(scene.arm_hand().compile())
    s.reset(arm_q=scene.ARM_HOME, hand_q=H.OPEN)
    quat = scene.hand_quat((0, 0, -1), (1, 0, 0))
    q = s.ik_high((0.45, -0.1, 0.5), quat)
    s.move(arm_q=q, duration=1.0)
    s.hold(0.3)
    assert min(s.d.body(b).xpos[2] for b in ("wrist_1_link", "wrist_2_link", "wrist_3_link")) > 0.5
    p0, p1 = s.site_pos("palm"), np.array([0.55, 0.1, 0.5])
    dev = []  # servo lag adds ~1 cm mid-move on a 28 cm path
    s.cart_move(p1, quat, 1.5, callback=lambda s, u: dev.append(
        np.linalg.norm(np.cross(s.site_pos("palm") - p0, (p1 - p0) / np.linalg.norm(p1 - p0)))))
    s.hold(0.3)
    assert max(dev) < 0.02 and np.linalg.norm(s.site_pos("palm") - p1) < 0.015  # arm servos sag ~1 cm under load


def test_tip_ik_moves_index_tip():
    s = Sim(scene.hand_only().compile())
    q0 = H.pose(index=(0, 0.8, 0.5, 0.3))
    s.reset(hand_q=q0)
    tip0 = s.site_pos("index_tip")
    q = s.tip_ik(q0, {"index": np.array([0, 0, -0.01])})
    s.reset(hand_q=q)
    assert np.linalg.norm(s.site_pos("index_tip") - tip0 - [0, 0, -0.01]) < 1e-3


def _max_self_penetration(m, qpos):
    """Deepest hand<->hand overlap (m) at qpos, checking every hand geom pair (not only the RL pair list)."""
    c = scene.hand_only().compile()  # default bitmasks: all hand pairs collide
    d = mujoco.MjData(c)
    d.qpos[:20] = qpos[:20]
    mujoco.mj_fwdPosition(c, d)
    return max([-k.dist for k in d.contact[: d.ncon]] + [0.0])


def test_rl_envs_block_finger_pass_through():
    from rl import baoding_env, cube_env, pen_env, rubik_env, screwdriver_env
    models = [cube_env.build_model(), pen_env.build_model(), screwdriver_env.build_model(), rubik_env.build_model(),
              baoding_env.build_model()[0]]
    crossed = [H.pose(thumb=(1.5, 0, 1.5, 1.5), index=(r, 2.0, 1.5, 1.5), middle=(-r, 2.0, 1.5, 1.5),
                      ring=(r, 2.0, 1.5, 1.5), pinky=(-r, 2.0, 1.5, 1.5)) for r in (0.6, -0.6)]
    for m in models:
        assert m.npair == len(scene.HAND_SELF_PAIRS)
        for q in crossed:
            s = Sim(m)
            s.reset(hand_q=H.OPEN)
            s.move_hand(q, 1.0)
            assert _max_self_penetration(m, s.d.qpos) < 0.004
