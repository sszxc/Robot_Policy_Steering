"""Multi-pose grasps with UR5e + HX5 hand: pick, lift 15 cm, hold 2 s, put back.

python -m tasks.grasps [--grasp power|pinch|tripod|lateral|hook] [--payload] [--video] [--viewer]
Each grasp is defined by the object's pose in the hand frame (palm +x, fingers +z, thumb +y).
"""
import argparse

import mujoco
import numpy as np

from sim import Sim, cli_sim, hand as H, scene

TABLE_Z = 0.4
SAFE_Z = TABLE_Z + 0.35
BASE_Z = 0.25  # UR5e on a pedestal so fingers-down poses over the table stay within reach
PALM_H = np.array([0.024, 0.0, 0.07])  # palm site in hand frame
CURL = (0, 1.6, 1.5, 1.5)  # finger tucked out of the way
DOWN_FWD = scene.hand_quat((0, 0, -1), (1, 0, 0))  # palm down, fingers forward
DOWN_OUT = scene.hand_quat((1, 0, 0), (0, 0, -1))  # fingers down, palm facing away from robot
START = np.array([2.723, -1.758, 1.518, 0.24, 1.153, 4.712])  # palm down at (0.5, 0, 0.8), wrist above hand

GRASPS = {
    # Side approach from -y: palm faces +y, fingers forward, thumb up; fingers wrap, thumb presses from wrist side.
    "power": dict(
        obj="bottle", xml="grasp_bottle.xml", pos=(0.5, -0.28, TABLE_Z + 0.09),
        quat=scene.hand_quat((0, 1, 0), (1, 0, 0)), p_h=(0.062, 0.0, 0.115), approach=(0, -0.12, 0), clear=0.2,
        pre=H.pose(thumb=(0, -1.4, 0, 0)),
        close=H.pose(thumb=(-0.5, -1.4, 1.0, 0.8), index=(0, 1.3, 1.5, 1.2), middle=(0, 1.3, 1.5, 1.2),
                     ring=(0, 1.3, 1.5, 1.2), pinky=(0, 1.3, 1.5, 1.2))),
    # Top approach: thumb and index tips squeeze a small cube along the hand z axis.
    "pinch": dict(
        obj="small_cube", xml="grasp_cube.xml", pos=(0.55, -0.12, TABLE_Z + 0.0175),
        quat=DOWN_FWD, p_h=(0.128, 0.036, 0.105), approach=(0, 0, 0.1),
        pre=H.pose(thumb=(0, -1.56, 0.35, 0), index=(0, 1.1, 0, 0), middle=CURL, ring=CURL, pinky=CURL),
        close=H.pose(thumb=(0, -1.56, 0.65, 0.1), index=(0, 1.75, 0, 0.1), middle=CURL, ring=CURL, pinky=CURL)),
    # Top approach: thumb below, index + middle above a ball.
    "tripod": dict(
        obj="ball", xml="grasp_ball.xml", pos=(0.55, 0.02, TABLE_Z + 0.022),
        quat=DOWN_FWD, p_h=(0.126, 0.0225, 0.108), approach=(0, 0, 0.1),
        pre=H.pose(thumb=(0.24, -1.57, 0.35, 0), index=(0, 1.05, 0, 0), middle=(0, 1.1, 0, 0), ring=CURL, pinky=CURL),
        close=H.pose(thumb=(0.24, -1.57, 0.7, 0.1), index=(0, 1.75, 0, 0.1), middle=(0, 1.8, 0, 0.1),
                     ring=CURL, pinky=CURL)),
    # Top approach: card in a slot pressed between thumb pad and the side of the curled index.
    "lateral": dict(
        obj="card", xml="grasp_card.xml", pos=(0.5, 0.15, TABLE_Z + 0.05),
        quat=DOWN_OUT, p_h=(0.061, 0.051, 0.167), approach=(0, 0, 0.1),
        pre=H.pose(thumb=(0.3, -0.36, 0.12, 0.6), index=(0, 1.0, 1.08, 1.0), middle=CURL, ring=CURL, pinky=CURL),
        close=H.pose(thumb=(1.15, -0.36, 0.12, 0.6), index=(0, 1.0, 1.08, 1.0), middle=CURL, ring=CURL, pinky=CURL)),
    # Top approach: straight fingers slide past the handle bar, then curl under it (no thumb).
    "hook": dict(
        obj="basket", xml="grasp_basket.xml", pos=(0.62, 0.3, TABLE_Z + 0.03),
        quat=DOWN_OUT, p_h=(0.042, -0.005, 0.29), approach=(0, 0, 0.1),  # bar at hand z 0.15, box 0.14 below
        pre=H.pose(thumb=(0, 0, 0, 0)),
        close=H.pose(index=(0, 0.3, 1.5, 1.4), middle=(0, 0.3, 1.5, 1.4), ring=(0, 0.3, 1.5, 1.4),
                     pinky=(0, 0.3, 1.5, 1.4))),
}


def build(names, mass=None):
    spec = scene.arm_hand()
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD  # several contacts on flat cylinder ends
    spec.body("base").pos = [0, 0, BASE_Z]
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.08, BASE_Z / 2], pos=[0, 0, BASE_Z / 2],
                            rgba=[0.3, 0.3, 0.32, 1])
    scene.add_table(spec, height=TABLE_Z)
    for n in names:
        g = GRASPS[n]
        if n == "lateral":
            scene.add_object(spec, "grasp_card_stand.xml", (g["pos"][0], g["pos"][1], TABLE_Z))
        scene.add_object(spec, g["xml"], g["pos"])
        if mass is not None:
            spec.geom(g["obj"]).mass = mass
    return spec.compile()


def palm_target(g):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(g["quat"], float))
    return np.asarray(g["pos"]) + R.reshape(3, 3) @ (PALM_H - np.asarray(g["p_h"]))


def ik_high(s, pos, quat):
    """IK over several seeds; keep the converged solution whose wrist stays highest (clear of the table)."""
    d = mujoco.MjData(s.m)
    rng = np.random.default_rng(0)
    seeds = [s.d.ctrl[:6].copy(), START, scene.ARM_HOME] + [START + rng.uniform(-1.5, 1.5, 6) for _ in range(20)]
    best, best_z = None, -np.inf
    for q0 in seeds:
        q = s.ik("palm", pos, quat, q0=q0)
        if s.ik_err > 1e-3:
            continue
        d.qpos[:6] = q
        mujoco.mj_kinematics(s.m, d)
        z = min(d.body(b).xpos[2] for b in ("wrist_1_link", "wrist_2_link", "wrist_3_link"))
        if z > best_z + 0.01:
            best, best_z = q, z
    return best


def cart_move(s, pos, quat, duration, hand_q=None, n=20):
    """Straight-line palm motion: IK on n waypoints, then follow them with a smooth time profile."""
    p0, q = s.site_pos("palm"), s.d.ctrl[:6].copy()
    path = [q]
    for a in np.linspace(0, 1, n + 1)[1:]:
        q = s.ik("palm", p0 + a * (np.asarray(pos) - p0), quat, q0=q)
        path.append(q)
    path = np.array(path)
    h0 = s.d.ctrl[s.hand_ctrl].copy()
    steps = max(1, int(duration / s.m.opt.timestep))
    for i in range(1, steps + 1):
        u = (0.5 - 0.5 * np.cos(np.pi * i / steps)) * n
        k = min(int(u), n - 1)
        s.d.ctrl[:6] = path[k] + (u - k) * (path[k + 1] - path[k])
        if hand_q is not None:
            s.d.ctrl[s.hand_ctrl] = h0 + (u / n) * (np.asarray(hand_q) - h0)
        s.step()


def run_grasp(s, name, place=True):
    """Execute one grasp; returns (success, object rise in m, hand links touching the object)."""
    g = GRASPS[name]
    target = palm_target(g)
    pre = target + np.asarray(g["approach"])
    above = pre + [0, 0, max(g.get("clear", 0.05), SAFE_Z - pre[2])]  # joint-space swings happen up here
    z0 = s.body_pos(g["obj"])[2]
    s.move(arm_q=ik_high(s, above, g["quat"]), hand_q=g["pre"], duration=2.0)
    cart_move(s, pre, g["quat"], 1.0)
    cart_move(s, target, g["quat"], 1.0)
    s.move_hand(g["close"], 0.8)
    s.hold(0.3)
    cart_move(s, target + [0, 0, 0.15], g["quat"], 1.2)
    s.hold(2.0)
    rise = s.body_pos(g["obj"])[2] - z0
    touch = sorted(s.hand_contacts(g["obj"]))
    ok = rise > 0.12 and len(touch) > 0
    if place:
        cart_move(s, target + [0, 0, 0.005], g["quat"], 1.2)
        s.hold(0.5)  # let the arm settle so the object is down before release
        s.move_hand(H.OPEN, 0.6)  # thumb back into the palm plane so nothing props the object up
        s.hold(0.4)
        cart_move(s, pre, g["quat"], 0.8)
        cart_move(s, above, g["quat"], 0.8)
    return ok, rise, touch


def payload(name, lo=0.05, hi=8.0, iters=8):
    """Bisection on object mass (kg) for the heaviest one still held after lift + 2 s hold."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = Sim(build([name], mass=mid))
        s.reset(arm_q=START, hand_q=H.OPEN)
        ok = run_grasp(s, name, place=False)[0]
        s.close()
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grasp", choices=list(GRASPS))
    p.add_argument("--payload", action="store_true")
    a, _ = p.parse_known_args()
    names = [a.grasp] if a.grasp else list(GRASPS)
    if a.payload:
        for n in (names if a.grasp else ["power", "pinch"]):
            print(f"{n:8s} max held mass ~ {payload(n):.2f} kg")
        return
    cam = dict(lookat=np.array([0.55, 0.0, 0.5]), distance=1.0, azimuth=150, elevation=-22)
    s = cli_sim(build(names), "grasps", cam=cam)
    s.reset(arm_q=START, hand_q=H.OPEN)
    s.hold(0.5)
    for n in names:
        s.cam.lookat[:] = np.asarray(GRASPS[n]["pos"]) + [0, 0, 0.12]  # video cuts to the current object
        ok, rise, touch = run_grasp(s, n)
        print(f"{n:8s} {'OK  ' if ok else 'FAIL'} rise={rise:.3f} m  contacts={touch}")
    s.close()


if __name__ == "__main__":
    main()
