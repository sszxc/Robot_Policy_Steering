"""Pick up a spray bottle (thumb + middle/ring/pinky wrap) and pull its trigger 3 times with the index finger.

Run: python -m tasks.spray [--viewer] [--video]
"""
import mujoco
import numpy as np

from sim import cli_sim, hand as H, scene

BOTTLE = np.array([0.45, -0.12, 0.4])      # bottle base on the table top
YAW = -np.radians(50)                      # trigger turned toward the fingertips so the index can hook it
NECK = BOTTLE + [0, 0, 0.232]              # grip point at index-finger height
GRIP_H = np.array([0.043, 0.036, 0.103])   # that point in the hand frame
PALM_H = np.array([0.024, 0.0, 0.07])      # palm site in the hand frame
HAND_R = np.column_stack([(1, 0, 0), (0, 0, 1), (0, -1, 0)]).astype(float)  # palm +x, thumb side up, fingers -y

PRESHAPE = H.pose(thumb=(0.75, -1.5, 0.3, 0.2), index=(0, 0.4, 0.2, 0.5), middle=(0, 0.3, 0.3, 0.3),
                  ring=(0, 0.3, 0.3, 0.3), pinky=(0, 0.3, 0.3, 0.3))
WRAP = (0, 1.6, 1.5, 1.5)  # overshoot targets: fingers stop on the grip and squeeze
GRASP = H.pose(thumb=(0.75, -1.5, 1.1, 1.0), index=(0, 0.7, 0.29, 1.03), middle=WRAP, ring=WRAP, pinky=WRAP)
SQUEEZE = GRASP.copy()
SQUEEZE[H.IDX["index"]] = (0, 0.8, 1.0, 1.35)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def palm_target(neck, R):
    """Palm-site pose that puts GRIP_H at the neck point with hand rotation R."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return neck - R @ (GRIP_H - PALM_H), q


def slerp(q0, q1, t):
    neg, dq, v, e, out = np.zeros(4), np.zeros(4), np.zeros(3), np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(neg, q0)
    mujoco.mju_mulQuat(dq, np.asarray(q1, float), neg)
    mujoco.mju_quat2Vel(v, dq, 1.0)
    angle = np.linalg.norm(v)
    if angle < 1e-9:
        return np.asarray(q1, float)
    mujoco.mju_axisAngle2Quat(e, v / angle, t * angle)
    mujoco.mju_mulQuat(out, e, q0)
    return out


def cart_move(s, pos, quat, duration, hand_q=None, n=20, callback=None):
    """Straight-line palm motion: IK on n waypoints, then one smooth pass through them."""
    p0 = s.site_pos("palm")
    q0 = np.zeros(4)
    mujoco.mju_mat2Quat(q0, s.d.site("palm").xmat)
    qs, q = [s.d.ctrl[:6].copy()], s.d.ctrl[:6].copy()
    for k in range(1, n + 1):
        q = s.ik("palm", p0 + (np.asarray(pos) - p0) * k / n, slerp(q0, quat, k / n), q0=q)
        qs.append(q)
    qs = np.array(qs)
    h0 = s.d.ctrl[s.hand_ctrl].copy()
    steps = max(1, int(duration / s.m.opt.timestep))
    for i in range(1, steps + 1):
        u = (0.5 - 0.5 * np.cos(np.pi * i / steps)) * n
        k = min(int(u), n - 1)
        s.d.ctrl[:6] = qs[k] + (u - k) * (qs[k + 1] - qs[k])
        if hand_q is not None:
            s.d.ctrl[s.hand_ctrl] = h0 + (u / n) * (np.asarray(hand_q) - h0)
        if callback:
            callback(s, u / n)
        s.step()


def in_palm(s):
    """Bottle position in the palm frame (to measure slip)."""
    return s.d.site("palm").xmat.reshape(3, 3).T @ (s.body_pos("spray") - s.site_pos("palm"))


def build():
    spec = scene.arm_hand()
    scene.add_table(spec)
    scene.add_object(spec, "spray.xml", BOTTLE, (np.cos(YAW / 2), 0, 0, np.sin(YAW / 2)))
    return spec.compile()


def main():
    m = build()
    cam = dict(lookat=np.array([0.42, -0.1, 0.66]), distance=0.8, azimuth=240, elevation=-15)
    s = cli_sim(m, "spray", cam)
    trig = m.joint("trigger")
    trig_max = m.jnt_range[trig.id][1]
    mist = m.geom("spray_mist").id
    s.reset(arm_q=scene.ARM_HOME, hand_q=H.OPEN)
    start = s.ik("palm", (0.3, 0.1, 0.8), scene.hand_quat((0, 0, -1), (1, 0, 0)), q0=scene.ARM_HOME)
    s.reset(arm_q=start, hand_q=H.OPEN)  # raised start: ARM_HOME would put the hand inside the bottle
    s.hold(0.3)

    peak = [0.0]

    def spray_fx(s, _):
        frac = np.clip(s.d.joint("trigger").qpos[0] / trig_max, 0, 1)
        peak[0] = max(peak[0], frac)
        s.m.geom_rgba[mist, 3] = 0.6 * max(0.0, (frac - 0.5) / 0.5)

    # Approach from behind the grip, then close.
    pos, quat = palm_target(NECK, HAND_R)
    s.move_site("palm", pos + [-0.15, 0, 0.15], quat, duration=1.5, hand_q=PRESHAPE)
    cart_move(s, pos + [-0.12, 0, 0], quat, 0.8)
    cart_move(s, pos, quat, 1.0)
    s.move_hand(GRASP, 0.8)
    s.hold(0.3)
    grasp_contacts = sorted(s.hand_contacts("spray"))

    # Lift, then turn to aim the nozzle to the side.
    rel0 = in_palm(s)
    cart_move(s, pos + [0, 0, 0.12], quat, 1.2)
    R_aim = rot_z(0.6) @ HAND_R
    pos_aim, quat_aim = palm_target(NECK + [-0.05, 0.05, 0.12], R_aim)
    cart_move(s, pos_aim, quat_aim, 1.2)
    s.hold(0.3)
    lifted = s.body_pos("spray")[2] - BOTTLE[2]

    peaks = []
    for _ in range(3):
        peak[0] = 0.0
        for target, dur in ((SQUEEZE, 0.35), (SQUEEZE, 0.2), (GRASP, 0.35), (GRASP, 0.15)):
            s.move(hand_q=target, duration=dur, callback=spray_fx)
        peaks.append(peak[0])
    s.m.geom_rgba[mist, 3] = 0
    rel1 = in_palm(s)
    held = s.body_pos("spray")[2] - BOTTLE[2]

    # Put it back, open fingers before the thumb (it sits in front of the neck), back straight out.
    cart_move(s, pos + [0, 0, 0.12], quat, 1.2)
    cart_move(s, pos + [0, 0, 0.003], quat, 1.0)
    fingers_open = GRASP.copy()
    fingers_open[4:] = PRESHAPE[4:]
    s.move_hand(fingers_open, 0.5)
    s.move_hand(PRESHAPE, 0.5)
    cart_move(s, pos + [-0.12, 0, 0], quat, 1.0)
    cart_move(s, pos + [-0.15, 0, 0.15], quat, 0.8)
    s.hold(0.5)
    placed = s.d.body("spray").xmat[8]  # cos(tilt)
    s.close()

    ok = lifted > 0.1 and held > 0.1 and min(peaks) > 0.8 and placed > np.cos(np.radians(10))
    print(f"grasp contacts: {grasp_contacts}")
    print(f"lift height {lifted:.3f} m, height after squeezes {held:.3f} m, "
          f"bottle slip in hand {np.linalg.norm(rel1 - rel0) * 1000:.1f} mm")
    print("trigger peaks (fraction of range):", np.round(peaks, 2))
    print(f"bottle tilt after release {np.degrees(np.arccos(min(1, placed))):.1f} deg")
    print("SUCCESS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
