"""Rubik's cube one-face turn: cube in a table fixture, UR5e holds the hand palm-down above it and the
thumb alone turns the top layer quarter by quarter (closed loop on the hinge angle).

python -m robotics.policies.scripted.rubik [--turns N] [--video] [--viewer]
Thumb stroke: lower at CMC C0, sweep CMC toward C1 (pushes a top-layer face), lift out, swing back.
"""
import argparse

import mujoco
import numpy as np

from robotics.sim import cli_sim, hand as H, scene

EDGE = 0.066          # cube edge
GAP = 0.0005          # clearance between top layer and lower body
DETENT = 0.01         # Nm, peak detent torque
COLORS = {"U": (1, 1, 1, 1), "D": (1, 0.85, 0, 1), "F": (0, 0.6, 0.2, 1), "B": (0, 0.3, 0.8, 1),
          "R": (0.8, 0.05, 0.05, 1), "L": (1, 0.45, 0, 1)}


def _box(body, pos, size, rgba, **kw):
    return body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), size=list(size), rgba=list(rgba), **kw)


def _stickers(body, z0, z1, zoff=0.0, top=False, bottom=False):
    """Visual 3x3 stickers on side faces for cube-frame heights z0..z1 (+top/bottom); zoff = body z in cube frame."""
    c, s, t = EDGE / 3, EDGE / 6 * 0.86, 0.0006
    h = EDGE / 2
    vis = dict(contype=0, conaffinity=0, density=0, group=2)
    rows = [z for z in (-c, 0, c) if z0 < z < z1]
    for i in (-c, 0, c):
        for z in rows:
            _box(body, (h, i, z - zoff), (t, s, s), COLORS["F"], **vis)
            _box(body, (-h, i, z - zoff), (t, s, s), COLORS["B"], **vis)
            _box(body, (i, h, z - zoff), (s, t, s), COLORS["L"], **vis)
            _box(body, (i, -h, z - zoff), (s, t, s), COLORS["R"], **vis)
        for j in (-c, 0, c):
            if top:
                _box(body, (i, j, h - zoff), (s, s, t), COLORS["U"], **vis)
            if bottom:
                _box(body, (i, j, -h), (s, s, t), COLORS["D"], **vis)


def add_rubik(spec, pos, quat=(1, 0, 0, 0), frictionloss=0.004, fixed=False, stickers=True):
    """Lower two layers ('rubik', free or fixed) + top layer ('rubik_top') on a hinge about the cube's +z axis."""
    h = EDGE / 2
    col = dict(condim=4, friction=[1, 0.01, 0.001])
    black = (0.08, 0.08, 0.08, 1)
    low = spec.worldbody.add_body(name="rubik", pos=list(pos), quat=list(quat))
    if not fixed:
        low.add_freejoint(name="rubik")
    _box(low, (0, 0, -h / 3), (h, h, 2 * h / 3), black, mass=0.06, **col)
    if stickers:
        _stickers(low, -h, h / 3, bottom=True)
    top = low.add_body(name="rubik_top", pos=[0, 0, 2 * h / 3])
    top.add_joint(name="rubik_top", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], frictionloss=frictionloss,
                  damping=0.0005, armature=1e-5)
    _box(top, (0, 0, 0), (h, h, h / 3 - GAP), black, mass=0.03, **col)
    if stickers:
        _stickers(top, h / 3, h, zoff=2 * h / 3, top=True)
    return low, top


def add_detent(s, peak=DETENT):
    """4-fold detent torque on the top-layer hinge (real cubes snap to quarter turns), applied every step."""
    qadr, dof = s.m.joint("rubik_top").qposadr[0], s.m.joint("rubik_top").dofadr[0]
    step = s.step

    def detent_step(n=1):
        for _ in range(n):
            s.d.qfrc_applied[dof] = -peak * np.sin(4 * s.d.qpos[qadr])
            step()
    s.step = detent_step
    return s


TABLE_Z = 0.4
CUBE = np.array([0.5, 0.035, 0.5])        # fixed cube center (world)
CUBE_IN_HAND = np.array([0.07 + EDGE / 2, 0.035, 0.07])  # cube center in hand frame (palm +x, fingers +z)
PALM_H = np.array([0.024, 0.0, 0.07])     # palm site in hand frame
QUAT = scene.hand_quat((0, 0, -1), (1, 0, 0))  # palm down, fingers +x
C0, C1 = -1.4, 1.0                         # thumb CMC sweep start/end
DOWN, UP = (-1.57, 0.8, 0.5), (-0.5, 0.3, 0.3)  # thumb (yaw, pitch, ip): pushing height / lifted out
SNAP = 40                                  # stop sweeping this many degrees before target; detent finishes


def thumb(cmc, yaw_pitch_ip):
    return H.pose(thumb=(cmc, *yaw_pitch_ip))


def palm_target():
    R = np.zeros(9); mujoco.mju_quat2Mat(R, QUAT); R = R.reshape(3, 3)
    return CUBE + R @ (PALM_H - CUBE_IN_HAND)


def build():
    spec = scene.arm_hand()
    scene.add_table(spec, pos=(0.55, 0, 0), height=TABLE_Z)
    bottom = CUBE[2] - EDGE / 2
    spec.worldbody.add_geom(name="fixture", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[*CUBE[:2], (TABLE_Z + bottom) / 2],
                            size=[0.02, 0.02, (bottom - TABLE_Z) / 2], rgba=[0.3, 0.3, 0.32, 1])
    add_rubik(spec, CUBE, fixed=True)
    return spec.compile()


def approach(s):
    """Lower the palm onto its target above the cube; the arm sags under the hand, so re-aim with the error."""
    tgt = palm_target()
    s.cart_move(tgt, QUAT, 1.5)
    for _ in range(2):
        s.hold(0.3)
        tgt = tgt + (palm_target() - s.site_pos("palm"))
        s.cart_move(tgt, QUAT, 0.5)
    s.hold(0.3)


def angle(s):
    return np.degrees(s.d.joint("rubik_top").qpos[0])


def quarter_turn(s, target, tries=3):
    """Thumb strokes until the layer settles within 10 deg of target (deg; negative = sweep direction).
    Returns strokes used (0 = failed)."""
    for i in range(tries):
        s.move_hand(thumb(C0, DOWN), 0.3)
        for c in np.arange(C0, C1 + 1e-6, 0.05):
            s.move_hand(thumb(c, DOWN), 0.035)
            if angle(s) <= target + SNAP:
                break
        s.move_hand(thumb(c, UP), 0.3)
        s.move_hand(thumb(C0, UP), 0.5)
        s.hold(0.2)
        if abs(angle(s) - target) < 10:
            return i + 1
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--turns", type=int, default=4)
    a, _ = p.parse_known_args()
    m = build()
    cam = dict(lookat=CUBE + [0, 0.02, 0.03], distance=0.4, azimuth=235, elevation=-22)
    s = add_detent(cli_sim(m, "rubik", cam))
    s.reset(arm_q=s.ik("palm", palm_target() + [0, 0, 0.1], QUAT, q0=scene.ARM_HOME), hand_q=thumb(C0, UP))
    s.hold(0.5)
    approach(s)
    print(f"palm error {np.linalg.norm(palm_target() - s.site_pos('palm')) * 1000:.1f} mm")
    strokes = []
    for k in range(a.turns):
        strokes.append(quarter_turn(s, -90 * (k + 1)))
        print(f"turn {k + 1}: layer angle {angle(s):7.1f} deg (target {-90 * (k + 1)})  strokes {strokes[-1]}")
    s.cart_move(palm_target() + [0, 0, 0.1], QUAT, 1.0)
    s.hold(0.5)
    s.close()
    print("SUCCESS" if all(strokes) else "FAIL")
    return all(strokes)


if __name__ == "__main__":
    main()
