"""Hammer a nail: pick the hammer off a rack with a power grasp, then drive a nail with wrist_3 swings.

Run: python -m tasks.hammer [--video] [--viewer]
"""
import mujoco
import numpy as np

from sim import cli_sim, hand as H, scene

TABLE_H = 0.4
RACK = np.array([0.5, -0.2, TABLE_H])      # rack base; hammer grip center sits 0.085 above it
NAIL = np.array([0.5, 0.25, TABLE_H])      # nail block base
PALM_DOWN = scene.hand_quat((0, 0, -1), (1, 0, 0))
# Palm site -> handle center (world, palm down) = handle at hand-frame (0.045, 0, 0.11); from a hold test.
GRIP_OFFSET = np.array([0.04, 0.0, -0.021])
FINGER = (0, 1.6, 1.5, 1.2)
GRASP = H.pose(thumb=(0.42, -1.35, 0.63, 1.26), index=FINGER, middle=FINGER, ring=FINGER, pinky=FINGER)
FIST = GRASP.copy()
FIST[H.IDX["thumb"]] = 0
# IK seed with wrist_2 above the hand (scene.ARM_HOME's wrist hangs below the palm and hits the rack).
WRIST_UP = np.array([2.4, -1.5, 1.5, 0.5, 1.2, -1.6])
TARGET_DEPTH = 0.02
WIND_UP, FOLLOW = 0.9, 0.5    # wrist_3 swing: lift angle and target past contact (rad)
SWING_T = 0.7                 # downswing time (s); peak ~pi rad/s, the UR5e wrist speed limit
WRIST_KV = 25                 # menagerie wrist servos (kv=100) are ~6x overdamped and lag badly on swings


def build():
    spec = scene.arm_hand()
    scene.add_table(spec, height=TABLE_H)
    scene.add_object(spec, "hammer_rack.xml", RACK)
    scene.add_object(spec, "hammer.xml", RACK + [0, 0, 0.085])
    scene.add_object(spec, "hammer_nail.xml", NAIL)
    m = spec.compile()
    for a in ("wrist_1", "wrist_2", "wrist_3"):
        m.actuator(a).biasprm[2] = -WRIST_KV
    return m


def in_hand_rot(s):
    return s.d.site("palm").xmat.reshape(3, 3).T @ s.d.body("hammer").xmat.reshape(3, 3)


def head_on_nail(s):
    return any({s.m.geom(c.geom1).name, s.m.geom(c.geom2).name} == {"hammer_head", "nail_head"}
               for c in s.d.contact[: s.d.ncon])


def strike(s, q):
    """Wind up wrist_3, swing down, and back off as soon as the head touches the nail. Returns face speed."""
    s.move(arm_q=q + [0, 0, 0, 0, 0, WIND_UP], duration=0.6)
    s.hold(0.2)
    c0, c1 = q[5] + WIND_UP, q[5] - FOLLOW
    n = int(SWING_T / s.m.opt.timestep)
    vel, speed = np.zeros(6), 0.0
    for i in range(1, n + 1):
        s.d.ctrl[5] = c0 + (c1 - c0) * (0.5 - 0.5 * np.cos(np.pi * i / n))
        s.step()
        if head_on_nail(s):
            break
        mujoco.mj_objectVelocity(s.m, s.d, mujoco.mjtObj.mjOBJ_SITE, s.m.site("hammer_face").id, vel, 0)
        speed = -vel[5]
    s.d.ctrl[5] = q[5] + 0.1
    s.hold(0.3)
    return speed


def main():
    m = build()
    s = cli_sim(m, "hammer", cam=dict(lookat=np.array([0.47, 0.05, 0.57]), distance=0.95, azimuth=165, elevation=-15))
    s.reset(arm_q=WRIST_UP, hand_q=H.OPEN)
    s.reset(arm_q=s.ik("palm", (0.5, 0, 0.6), PALM_DOWN), hand_q=H.OPEN)
    s.hold(0.3)
    nail_q = m.joint("nail").qposadr[0]

    # Grasp the handle from above and lift.
    grip = s.body_pos("hammer")
    palm = grip - GRIP_OFFSET
    s.move_site("palm", palm + [0, 0, 0.12], PALM_DOWN, duration=1.5)
    s.move_site("palm", palm, PALM_DOWN, duration=1.0)
    s.hold(0.2)
    s.move_hand(FIST, 0.8)
    s.move_hand(GRASP, 0.8)
    s.hold(0.3)
    s.move_site("palm", palm + [0, 0, 0.15], PALM_DOWN, duration=1.0)
    R0 = in_hand_rot(s)

    # Place the head above the nail using the actual in-hand hammer pose.
    face_to_palm = s.site_pos("palm") - s.site_pos("hammer_face")
    head = s.body_pos("nail") + [0, 0, 0.002]
    s.move_site("palm", head + face_to_palm + [0, 0, 0.12], PALM_DOWN, duration=1.5)
    s.hold(0.3)
    face_to_palm = s.site_pos("palm") - s.site_pos("hammer_face")
    hover = head + face_to_palm + [0, 0, 0.01]  # face 1 cm above the nail head

    strikes = 0
    while s.d.qpos[nail_q] < TARGET_DEPTH and strikes < 15:
        # Hover (following the nail down), wind up, swing through, back off.
        q = s.move_site("palm", hover - [0, 0, s.d.qpos[nail_q]], PALM_DOWN, duration=0.6)
        s.hold(0.2)
        speed = strike(s, q)
        strikes += 1
        print(f"strike {strikes}: head speed {speed:.2f} m/s, nail depth {s.d.qpos[nail_q] * 100:.1f} cm, "
              f"hand contacts {len(s.hand_contacts('hammer'))}")

    s.move_site("palm", s.site_pos("palm") + [0, 0, 0.1], PALM_DOWN, duration=1.0)
    s.hold(0.5)
    held = s.body_pos("hammer")[2] > TABLE_H + 0.15 and len(s.hand_contacts("hammer")) > 0
    depth = s.d.qpos[nail_q]
    slip = np.degrees(np.arccos(np.clip((np.trace(R0.T @ in_hand_rot(s)) - 1) / 2, -1, 1)))
    print(f"depth {depth * 100:.1f} cm in {strikes} strikes, hammer held: {held}, "
          f"in-hand rotation {slip:.1f} deg, success: {depth >= TARGET_DEPTH and held}")
    s.close()
    return depth, held


if __name__ == "__main__":
    main()
