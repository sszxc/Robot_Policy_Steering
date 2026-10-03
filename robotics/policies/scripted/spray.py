"""Pick up a spray bottle (thumb + middle/ring/pinky wrap) and pull its trigger 3 times with the index finger.

Run: python -m robotics.policies.scripted.spray [--viewer] [--video]
"""
import mujoco
import numpy as np

from robotics.sim import cli_sim, hand as H, scene

BOTTLE = np.array([0.45, -0.12, 0.4])      # bottle base on the table top
YAW = -np.radians(50)                      # trigger turned toward the fingertips so the index can hook it
NECK = BOTTLE + [0, 0, 0.232]              # grip point at index-finger height
GRIP_H = np.array([0.043, 0.036, 0.103])   # that point in the hand frame
PALM_H = np.array([0.024, 0.0, 0.07])      # palm site in the hand frame
HAND_R = np.column_stack([(1, 0, 0), (0, 0, 1), (0, -1, 0)]).astype(float)  # palm +x, thumb side up, fingers -y

PRESHAPE = H.pose(thumb=(0.75, -1.5, 0.3, 0.2), index=(0, 0.4, 0.2, 0.5), middle=(0, 0.3, 0.3, 0.3),
                  ring=(0, 0.3, 0.3, 0.3), pinky=(0, 0.3, 0.3, 0.3))
GRASP = H.pose(thumb=(0.75, -1.5, 1.1, 1.0), index=(0, 0.7, 0.29, 1.03), middle=H.CURL, ring=H.CURL, pinky=H.CURL)
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
    start = s.ik_high((0.3, 0.1, 0.8), scene.hand_quat((0, 0, -1), (1, 0, 0)))
    s.reset(arm_q=start, hand_q=H.OPEN)  # raised start: ARM_HOME would put the hand inside the bottle
    s.hold(0.3)

    peak = [0.0]

    def spray_fx(s, _):
        frac = np.clip(s.d.joint("trigger").qpos[0] / trig_max, 0, 1)
        peak[0] = max(peak[0], frac)
        s.m.geom_rgba[mist, 3] = 0.6 * max(0.0, (frac - 0.5) / 0.5)

    # Approach from behind the grip, then close.
    pos, quat = palm_target(NECK, HAND_R)
    s.move(arm_q=s.ik_high(pos + [-0.15, 0, 0.15], quat), hand_q=PRESHAPE, duration=1.5)
    s.cart_move(pos + [-0.12, 0, 0], quat, 0.8)
    s.cart_move(pos, quat, 1.0)
    s.move_hand(GRASP, 0.8)
    s.hold(0.3)
    grasp_contacts = sorted(s.hand_contacts("spray"))

    # Lift, then turn to aim the nozzle to the side.
    rel0 = in_palm(s)
    s.cart_move(pos + [0, 0, 0.12], quat, 1.2)
    R_aim = rot_z(0.6) @ HAND_R
    pos_aim, quat_aim = palm_target(NECK + [-0.05, 0.05, 0.12], R_aim)
    s.cart_move(pos_aim, quat_aim, 1.2)
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
    s.cart_move(pos + [0, 0, 0.12], quat, 1.2)
    s.cart_move(pos + [0, 0, 0.003], quat, 1.0)
    fingers_open = GRASP.copy()
    fingers_open[4:] = PRESHAPE[4:]
    s.move_hand(fingers_open, 0.5)
    s.move_hand(PRESHAPE, 0.5)
    s.cart_move(pos + [-0.12, 0, 0], quat, 1.0)
    s.cart_move(pos + [-0.15, 0, 0.15], quat, 0.8)
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
