"""Screwdriver twirl with fingers only (finger gaiting): the index sweeps the handle surface tangentially, lifts off,
returns and re-touches; thread friction holds the handle meanwhile. Wrist fixed, palm down. python -m robotics.policies.scripted.screwdriver"""
import mujoco
import numpy as np

from robotics.sim import cli_sim, hand as H, scene

C = np.array([0.114, 0.030])  # handle axis (x, y)
ZC = 0.198                     # fingertip contact height (handle spans z 0.13..0.21)
R = 0.015                      # handle radius
BASE = H.pose(index=(0, 1.2, 0.6, 0.4), middle=(0, 0.3, 0.3, 0.2))
# Per finger: handle side (+1 = +x) and surface arc swept (y offsets from the axis, m), clockwise torque.
# A thumb sweep on the -x side (-1, -0.008, 0.008) at phase 0.5 was tried: +8 deg/cycle alone, and it slowed the index.
SWEEP = {"index": (1, 0.0105, -0.008)}
OFFSET = {"index": 0.0}  # gait phase offsets
# Thumb presses the -x side while the index is off, so the hand never loses contact (~22 vs ~38 deg/cycle).
THUMB_BRAKE = False
if THUMB_BRAKE:
    SWEEP["thumb"], OFFSET["thumb"] = (-1, 0.0, 0.0), 0.45


def build():
    spec = scene.hand_only(pos=(0, 0, 0.3), quat=scene.hand_quat((0, 0, -1), (1, 0, 0)))
    scene.add_object(spec, "screwdriver.xml", pos=(C[0], C[1], 0.13))
    spec.worldbody.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.06, 0.01],
                            pos=[C[0], C[1], 0.02], rgba=[0.55, 0.4, 0.28, 1], contype=0, conaffinity=0)
    return spec


def tips_at(s, q):
    d = mujoco.MjData(s.m)
    d.qpos[:] = s.d.qpos
    for i, j in enumerate(s.m.actuator_trnid[s.hand_ctrl, 0]):
        d.qpos[s.m.jnt_qposadr[j]] = q[i]
    mujoco.mj_kinematics(s.m, d)
    return {f: d.site(f + "_tip").xpos.copy() for f in H.FINGERS}


def reach(s, targets, q0=BASE, n=10):
    """Hand pose putting fingertip sites at world targets ({finger: xyz}); walks there in n small IK steps from q0."""
    q = np.array(q0, float)
    t0 = tips_at(s, q)
    for k in range(1, n + 1):
        cur = tips_at(s, q)
        q = s.tip_ik(q, {f: t0[f] + (np.asarray(p) - t0[f]) * k / n - cur[f] for f, p in targets.items()},
                     lock_roll=False)
    return q


def angle(s):
    return s.d.joint("sd_turn").qpos[0]


def finger_loop(s, f, squeeze=0.004, lift=0.012, n=6):
    """Joint targets (4,) along one gait loop: n points sweeping the surface arc, then lift, return, re-touch.
    Returns (phases, poses): phase in [0,1) at which each pose is reached."""
    side, u0, u1 = SWEEP[f]
    pts = [(side * np.sqrt((R - squeeze) ** 2 - u ** 2), u) for u in np.linspace(u0, u1, n)]
    pts += [(side * (np.sqrt(R ** 2 - u1 ** 2) + lift), u1), (side * (np.sqrt(R ** 2 - u0 ** 2) + lift), u0)]
    phases = list(np.linspace(0, 0.5, n)) + [0.6, 0.85]
    poses, q = [], BASE
    for x, u in pts:
        q = reach(s, {f: (C[0] + x, C[1] + u, ZC)}, q)
        poses.append(q[H.IDX[f]])
    return np.array(phases + [1.0]), np.array(poses + [poses[0]])


def gait_pose(loops, t, period):
    """Hand targets at time t: each finger interpolates its loop, shifted by its phase offset."""
    q = BASE.copy()
    for f, (ph, poses) in loops.items():
        p = (t / period + OFFSET[f]) % 1.0
        q[H.IDX[f]] = [np.interp(p, ph, poses[:, j]) for j in range(4)]
    return q


def main(cycles=12, period=1.2):
    s = cli_sim(build().compile(), "screwdriver",
                cam=dict(lookat=np.array([0.1, 0.02, 0.21]), distance=0.42, azimuth=145, elevation=-22))
    loops = {f: finger_loop(s, f, squeeze=0.002 if f == "thumb" else 0.004) for f in SWEEP}
    s.reset(hand_q=BASE)
    s.move_hand(gait_pose(loops, 0.0, period), 0.5)
    s.hold(0.2)
    a_start, per = angle(s), []
    for c in range(cycles):
        a0 = angle(s)
        for i in range(int(period / s.m.opt.timestep)):
            s.d.ctrl[s.hand_ctrl] = gait_pose(loops, i * s.m.opt.timestep, period)
            s.step()
        per.append(np.degrees(a0 - angle(s)))
        print(f"cycle {c:2d}: {per[-1]:+6.1f} deg   total {np.degrees(a_start - angle(s)):6.0f} deg   "
              f"contacts {sorted(s.hand_contacts('screwdriver'))}")
    total = np.degrees(a_start - angle(s))
    print(f"total clockwise rotation {total:.0f} deg in {cycles} cycles ({total / cycles:.1f} deg/cycle); "
          f"success: {total >= 360}")
    s.close()
    return total


if __name__ == "__main__":
    main()
