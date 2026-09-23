"""In-hand alignment of a rod held in a thumb-index-middle pinch (UR5e + HX5, privileged rod pose).

Demos (each from a fresh grasp): (a) slide the rod down through the pinch with gravity using grip pulses,
(b) pivot it by differential index/middle pushes, (c) translate it with all three fingertips (Jacobian IK).
Run: python -m tasks.align [--video] [--viewer]
"""
import mujoco
import numpy as np

from sim import cli_sim, hand as H, scene

CURL = (0, 0.3, 0.3, 0.2)  # ring/pinky kept clear of the rod
# Tripod pinch found by optimizing thumb/index/middle pad opposition around a 12 mm rod.
GRIP = H.pose(thumb=(0.196, -1.571, 0.491, 0.0), index=(0, 1.66, 0, 0), middle=(0, 1.804, 0, 0), ring=CURL, pinky=CURL)
SQUEEZE = {"thumb": (2, 3), "index": (5,), "middle": (9,)}  # joints pushed past contact to create grip force
TIPS = ("thumb", "index", "middle")
PALM_POS = (0.5, 0.0, 0.45)
CAM = dict(lookat=(0.52, 0.08, 0.46), distance=0.36, azimuth=-60, elevation=-12)


def gripped(q, thumb=0.15, index=0.15, middle=0.15):
    q = np.array(q, float)
    for f, amt in (("thumb", thumb), ("index", index), ("middle", middle)):
        for i in SQUEEZE[f]:
            q[i] += amt
    return q


class Align:
    def __init__(self, s):
        self.s, self.m, self.d = s, s.m, s.d
        self.q = GRIP.copy()  # hand pose without squeeze

    # ---- measurements (world frame) ----
    def pinch(self):
        return 0.5 * (self.s.site_pos("thumb_tip") + self.s.site_pos("index_tip"))

    def rod_axis(self):
        return self.d.body("rod").xmat.reshape(3, 3)[:, 2].copy()

    def offset(self):
        """Rod COM position along its own axis relative to the pinch point (m)."""
        return (self.s.body_pos("rod") - self.pinch()) @ self.rod_axis()

    def hand_R(self):
        return self.d.body("hand_base").xmat.reshape(3, 3).copy()

    def rod_axis_in_hand(self):
        return self.hand_R().T @ self.rod_axis()

    def rod_in_hand(self):
        return self.hand_R().T @ (self.s.body_pos("rod") - self.d.body("hand_base").xpos)

    # ---- actions ----
    def set_grip(self, **amt):
        self.d.ctrl[self.s.hand_ctrl] = gripped(self.q, **amt)

    def slip_to(self, err, step=(4e-4, 1.2e-3), fingers=TIPS, hold=0.04, max_pulses=200):
        """Grip pulses: briefly drop the squeeze of `fingers` so the rod creeps under gravity.
        Pulse depth adapts so each pulse moves err() by step[0]..step[1]; stops when err() crosses zero."""
        pulse, e0, moved = hold, np.sign(err()), 0.0
        for _ in range(max_pulses):
            e = err()
            if np.sign(e) != e0 or abs(e) < 0.5 * moved:
                break
            self.set_grip(**{f: pulse if f in fingers else hold for f in TIPS})
            self.s.hold(0.02)
            self.set_grip(thumb=hold, index=hold, middle=hold)
            self.s.hold(0.06)
            moved = abs(err() - e)
            lo, hi = min(step[0], abs(e) / 6), min(step[1], abs(e) / 3)  # smaller steps near the goal
            pulse += -0.001 if moved < lo else (0.006 if moved > hi else 0.0)
        self.set_grip()
        self.s.hold(0.3)

    def tip_ik(self, delta):
        """Hand pose moving the thumb/index/middle tips by `delta` (world vector, or dict per finger) from self.q."""
        delta = delta if isinstance(delta, dict) else {f: delta for f in TIPS}
        return self.s.tip_ik(self.q, delta)

def setup(s):
    """Arm in handshake pose (palm +y, fingers +x, thumb up), rod placed in the pinch, grip closed."""
    arm_q = s.ik("palm", PALM_POS, scene.hand_quat((0, 1, 0), (1, 0, 0)), q0=scene.ARM_HOME)
    s.reset(arm_q=arm_q, hand_q=GRIP * 0.85)
    pt, it, mt = (s.site_pos(f + "_tip") for f in TIPS)
    tn = s.d.body("finger_r_link4").xmat.reshape(3, 3)[:, 2]
    axis = it - mt
    axis -= tn * (axis @ tn)
    quat = np.zeros(4)
    mujoco.mju_quatZ2Vec(quat, axis / np.linalg.norm(axis))
    j = s.m.joint("rod").qposadr[0]
    s.d.qpos[j:j + 3] = pt + 0.0085 * tn
    s.d.qpos[j + 3:j + 7] = quat
    # close the grip with gravity off so the rod starts centered in the pinch
    g = s.m.opt.gravity.copy()
    s.m.opt.gravity[:] = 0
    s.move_hand(gripped(GRIP), 0.4)
    s.m.opt.gravity[:] = g
    s.hold(0.5)
    return arm_q


def main():
    spec = scene.arm_hand()
    scene.add_object(spec, "align_rod.xml", (0.6, 0, 0.8))
    s = cli_sim(spec.compile(), "align", cam=CAM)
    res = {}

    # (a) slide: with the thumb up gravity pulls the rod down through the pinch (COM stays below it)
    setup(s)
    a = Align(s)
    off0 = a.offset()
    for d in (-0.01, -0.02):  # beyond ~25 mm the rod tilts in the pinch and slips run away
        a.slip_to(lambda: off0 + d - a.offset())
        res[f"slide {d * 100:+.1f} cm"] = a.offset() - off0 - d
        print(f"(a) slide      target {d * 1000:+.1f} mm   achieved {(a.offset() - off0) * 1000:+.1f} mm")

    # (b) pivot (fresh grasp): index pushes toward the thumb while middle backs off, rolling the rod
    # about the thumb contact; closed loop on the rod angle in the hand frame
    setup(s)
    a = Align(s)
    R = a.hand_R()
    u0 = a.rod_axis_in_hand()
    tn = R.T @ (s.site_pos("index_tip") - s.site_pos("thumb_tip"))
    tn /= np.linalg.norm(tn)
    k = np.cross(tn, u0)
    k /= np.linalg.norm(k)
    angle = lambda: np.degrees(np.arctan2(np.cross(u0, a.rod_axis_in_hand()) @ k, u0 @ a.rod_axis_in_hand()))
    push = 0.0
    for tgt in (8.0, 16.0):  # one-sided: index extension hits joint limits, reversing tends to drop the rod
        for _ in range(6):
            push = float(np.clip(push + 0.00012 * (tgt - angle()), -0.005, 0.005))
            a.q = GRIP.copy()
            a.q = a.tip_ik({"thumb": np.zeros(3), "index": -push * R @ tn, "middle": push * R @ tn})
            s.move(hand_q=gripped(a.q), duration=0.4)
            s.hold(0.1)
        res[f"pivot {tgt:+.0f} deg"] = angle() - tgt
        print(f"(b) pivot      target {tgt:+.1f} deg  achieved {angle():+.1f} deg")
    s.hold(0.3)

    # (c) translate (fresh grasp) across the rod axis by moving all three fingertips together
    setup(s)
    a = Align(s)
    for delta in ((0, 0, -0.006), (0, 0, 0.006)):
        p0 = a.rod_in_hand()
        goal = p0 + np.array(delta)
        for _ in range(4):
            u = a.rod_axis_in_hand()
            err = goal - a.rod_in_hand()
            err -= u * (err @ u)  # axial slip is not controlled here
            a.q = a.tip_ik(a.hand_R() @ np.clip(0.8 * err, -0.006, 0.006))
            s.move(hand_q=gripped(a.q), duration=0.4)
            s.hold(0.1)
        e = a.rod_in_hand() - goal
        e -= a.rod_axis_in_hand() * (e @ a.rod_axis_in_hand())
        res[f"translate {delta[2] * 1000:+.0f} mm"] = np.linalg.norm(e)
        print(f"(c) translate  target {delta[2] * 1000:+.1f} mm   achieved {(a.rod_in_hand() - p0)[2] * 1000:+.1f} mm"
              f"   (error across rod axis {np.linalg.norm(e) * 1000:.1f} mm)")
    s.hold(0.5)
    s.close()
    return res


if __name__ == "__main__":
    main()
