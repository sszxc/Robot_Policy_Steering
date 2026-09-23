"""Scissors: thumb and index go through the finger loops; open/close by finger motion only.

Hand palm faces sideways (thumb up, fingers forward). The thumb CMC swings the thumb loop away
from the static index loop to open. First close snips a paper sheet (weld released, half falls).
"""
import mujoco
import numpy as np

from sim import cli_sim, hand as H, scene

HAND_POS = (0, 0, 0.25)
TILT = np.radians(12)  # palm faces sideways, tilted up so gravity seats the loops on the fingers
HAND_QUAT = scene.hand_quat((0, np.cos(TILT), np.sin(TILT)), (1, 0, 0))
CURL = (0, 1.4, 1.5, 1.2)
CLOSED = H.pose(thumb=(0.5, -1.0, 0, 0), index=(-0.2, 0.7, 0.7, 0.2), middle=CURL, ring=CURL, pinky=CURL)
OPEN = H.pose(thumb=(-0.8, -1.0, 0, 0), index=(-0.2, 0.7, 0.7, 0.2), middle=CURL, ring=CURL, pinky=CURL)

HINGE_OFS = np.array([0.09, 0.11, 0.19])  # pivot in hand frame, above/ahead of the loops (handles form a V)
RING_R, WIRE_R = 0.0185, 0.0035           # loop centerline radius, wire radius
BLADE_LEN, BLADE_W, BLADE_T = 0.09, 0.016, 0.002
PAPER = (0.028, 0.03, 0.0005)  # half sizes: along blade, across blade line, thickness
OPEN_RANGE = 0.8                          # hinge range [0, OPEN_RANGE] rad
CYCLES = 5
DENSITY = 2700  # aluminium blades keep the CoM near the loops
THUMB_LINK, THUMB_AT = "finger_r_link4", 0.02
INDEX_LINK, INDEX_AT = "finger_r_link8", 0.014


def _loops(m, d):
    """World centers/axes of the thumb and index phalanges the loops sit on."""
    b = d.body(THUMB_LINK); R = b.xmat.reshape(3, 3)
    T, ta = b.xpos + R @ [0, THUMB_AT, 0], R[:, 1]
    b = d.body(INDEX_LINK); R = b.xmat.reshape(3, 3)
    F, fa = b.xpos + R @ [0, 0, INDEX_AT], R[:, 2]
    return T, ta, F, fa


def _ring(body, center, axis, rgba, n=10):
    u = np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    pts = [center + RING_R * (np.cos(a) * u + np.sin(a) * v) for a in np.linspace(0, 2 * np.pi, n + 1)]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*p0, *p1], size=[WIRE_R, 0, 0], rgba=rgba,
                      density=1100, condim=4, friction=[1, 0.01, 0.001])
    return pts


def _blade(body, b, s, side, rgba):
    """Thin box from the pivot along b, offset along the hinge axis s so the blades don't overlap."""
    mat = np.column_stack([b, s, np.cross(b, s)])
    q = np.zeros(4); mujoco.mju_mat2Quat(q, mat.ravel())
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=b * BLADE_LEN / 2 + s * side * BLADE_T,
                  quat=q, size=[BLADE_LEN / 2, BLADE_T / 2, BLADE_W / 2], rgba=rgba, density=DENSITY)


def _add_paper(spec, P, b, s):
    """Paper sheet across the blade line: fixed half + half welded to it; the weld is removed on the snip."""
    mat = np.column_stack([b, s, np.cross(b, s)])
    q = np.zeros(4); mujoco.mju_mat2Quat(q, mat.ravel())
    center = P + b * (0.03 + PAPER[0])
    for name, side in (("paper_fixed", -1), ("paper_cut", 1)):
        body = spec.worldbody.add_body(name=name, pos=center + s * side * PAPER[1], quat=q)
        if side > 0:
            body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=PAPER, rgba=[0.95, 0.95, 0.9, 1], density=800)
        for other in ("scissors", "scissors_b"):
            spec.add_exclude(bodyname1=name, bodyname2=other)
    spec.add_exclude(bodyname1="paper_fixed", bodyname2="paper_cut")
    spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name="paper", name1="paper_cut", name2="paper_fixed",
                      objtype=mujoco.mjtObj.mjOBJ_BODY, data=[0] * 10 + [1])


def build():
    """Hand-only scene with the scissors placed (closed) around thumb and index at the CLOSED posture."""
    spec = scene.hand_only(pos=HAND_POS, quat=HAND_QUAT)
    m = spec.compile(); d = mujoco.MjData(m)
    d.qpos[:] = CLOSED; mujoco.mj_kinematics(m, d)
    T, ta, F, fa = _loops(m, d)
    Rh = d.body("hand_base").xmat.reshape(3, 3)
    P = d.body("hand_base").xpos + Rh @ HINGE_OFS
    s = np.cross(T - P, F - P); s /= np.linalg.norm(s)  # +rotation of the index piece opens
    b = (P - T) / np.linalg.norm(P - T) + (P - F) / np.linalg.norm(P - F)
    b /= np.linalg.norm(b)  # closed blades continue the V of the handles

    body_a = spec.worldbody.add_body(name="scissors", pos=P)
    body_a.add_freejoint(name="scissors")
    body_b = body_a.add_body(name="scissors_b")
    # scene_base.xml has no compiler angle="radian", so spec-added joint ranges are in degrees
    body_b.add_joint(name="scissors_hinge", axis=s, range=[0, np.degrees(OPEN_RANGE)], damping=0.002)
    for body, center, axis, side, rgba in ((body_a, T, ta, 1, [0.9, 0.2, 0.2, 1]), (body_b, F, fa, -1, [0.2, 0.4, 0.9, 1])):
        pts = np.array(_ring(body, center - P, axis, rgba))
        edge = pts[np.argmin(np.linalg.norm(pts, axis=1))]  # ring point nearest the pivot
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*(s * side * BLADE_T), *edge],
                      size=[WIRE_R, 0, 0], rgba=rgba, density=1100, condim=4)
        _blade(body, b, s, side, [0.8, 0.8, 0.85, 1])
    body_a.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[*(-2.5 * BLADE_T * s), *(2.5 * BLADE_T * s)],
                    size=[0.005, 0, 0], rgba=[0.3, 0.3, 0.3, 1], contype=0, conaffinity=0, density=0)
    # let the scissors settle in the closed hand, then put the paper on the settled blade line
    m = spec.compile(); d = mujoco.MjData(m)
    d.qpos[:20] = d.ctrl[:] = CLOSED
    mujoco.mj_step(m, d, 250)
    R = d.body("scissors").xmat.reshape(3, 3)
    _add_paper(spec, d.body("scissors").xpos, R @ b, R @ s)
    return spec.compile()


def _snip(sim, _):
    """Release the paper weld once the blades have closed on it."""
    if sim.d.qpos[sim.m.joint("scissors_hinge").qposadr[0]] < 0.08:
        sim.d.eq_active[sim.m.equality("paper").id] = 0


def main():
    m = build()
    cam = dict(lookat=np.array([0.15, 0.07, 0.29]), distance=0.4, azimuth=-145, elevation=-20)
    sim = cli_sim(m, "scissors", cam)
    sim.reset(hand_q=CLOSED)
    hinge = m.joint("scissors_hinge").qposadr[0]
    sim.hold(0.5)
    z0 = sim.body_pos("scissors")[2]
    paper_z = sim.body_pos("paper_cut")[2]
    lo, hi = [], []
    for i in range(CYCLES):
        sim.move_hand(OPEN, 0.8); sim.hold(0.2); hi.append(sim.d.qpos[hinge])
        sim.move(hand_q=CLOSED, duration=0.8, callback=_snip if i == 0 else None); sim.hold(0.2)
        lo.append(sim.d.qpos[hinge])
    swing = (np.array(hi) - np.array(lo)) / OPEN_RANGE
    thumb, index = {f"finger_r_link{i}" for i in (3, 4)}, {f"finger_r_link{i}" for i in (6, 7, 8)}
    held = bool(thumb & sim.hand_contacts("scissors")) and bool(index & sim.hand_contacts("scissors_b"))
    drop = z0 - sim.body_pos("scissors")[2]
    cut = paper_z - sim.body_pos("paper_cut")[2] > 0.05
    print("open angles [rad]:", np.round(hi, 3), " closed:", np.round(lo, 3))
    print(f"swing per cycle (fraction of range): {np.round(swing, 2)}  loops on fingers: {held}  pivot drop: {drop * 1000:.1f} mm  paper cut: {cut}")
    ok = swing.min() >= 0.7 and held and drop < 0.03
    print("SUCCESS" if ok else "FAIL")
    sim.close()
    return ok


if __name__ == "__main__":
    main()
