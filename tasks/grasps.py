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
DOWN_FWD = scene.hand_quat((0, 0, -1), (1, 0, 0))  # palm down, fingers forward
DOWN_OUT = scene.hand_quat((1, 0, 0), (0, 0, -1))  # fingers down, palm facing away from robot
START = np.array([2.723, -1.758, 1.518, 0.24, 1.153, 4.712])  # palm down at (0.5, 0, 0.8), wrist above hand

GRASPS = {
    # Side approach from -y: palm faces +y, fingers forward, thumb up; fingers wrap, thumb presses from wrist side.
    "power": dict(
        obj="bottle", xml="grasp_bottle.xml", pos=(0.5, -0.28, TABLE_Z + 0.09),
        quat=scene.hand_quat((0, 1, 0), (1, 0, 0)), p_h=(0.062, 0.0, 0.115), approach=(0, -0.12, 0), clear=0.2,
        pre=H.pose(thumb=(0, -1.4, 0, 0)), close=H.POWER),
    # Top approach: thumb and index tips squeeze a small cube along the hand z axis.
    "pinch": dict(
        obj="small_cube", xml="grasp_cube.xml", pos=(0.55, -0.12, TABLE_Z + 0.0175),
        quat=DOWN_FWD, p_h=(0.128, 0.036, 0.105), approach=(0, 0, 0.1),
        pre=H.pose(thumb=(0, -1.56, 0.35, 0), index=(0, 1.1, 0, 0), middle=H.CURL, ring=H.CURL, pinky=H.CURL),
        close=H.PINCH),
    # Top approach: thumb below, index + middle above a ball.
    "tripod": dict(
        obj="ball", xml="grasp_ball.xml", pos=(0.55, 0.02, TABLE_Z + 0.022),
        quat=DOWN_FWD, p_h=(0.126, 0.0225, 0.108), approach=(0, 0, 0.1),
        pre=H.pose(thumb=(0.24, -1.57, 0.35, 0), index=(0, 1.05, 0, 0), middle=(0, 1.1, 0, 0), ring=H.CURL, pinky=H.CURL),
        close=H.TRIPOD),
    # Top approach: card in a slot pressed between thumb pad and the side of the curled index.
    "lateral": dict(
        obj="card", xml="grasp_card.xml", pos=(0.5, 0.15, TABLE_Z + 0.05),
        quat=DOWN_OUT, p_h=(0.061, 0.051, 0.167), approach=(0, 0, 0.1),
        pre=H.pose(thumb=(0.3, -0.36, 0.12, 0.6), index=(0, 1.0, 1.08, 1.0), middle=H.CURL, ring=H.CURL, pinky=H.CURL),
        close=H.KEY_PINCH),
    # Top approach: straight fingers slide past the handle bar, then curl under it (no thumb).
    "hook": dict(
        obj="basket", xml="grasp_basket.xml", pos=(0.62, 0.3, TABLE_Z + 0.03),
        quat=DOWN_OUT, p_h=(0.042, -0.005, 0.29), approach=(0, 0, 0.1),  # bar at hand z 0.15, box 0.14 below
        pre=H.pose(thumb=(0, 0, 0, 0)),
        close=H.HOOK),
}


def build(names, mass=None):
    spec = scene.arm_hand(base_height=BASE_Z)
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


def run_grasp(s, name, place=True):
    """Execute one grasp; returns (success, object rise in m, hand links touching the object)."""
    g = GRASPS[name]
    target = palm_target(g)
    pre = target + np.asarray(g["approach"])
    above = pre + [0, 0, max(g.get("clear", 0.05), SAFE_Z - pre[2])]  # joint-space swings happen up here
    z0 = s.body_pos(g["obj"])[2]
    s.move(arm_q=s.ik_high(above, g["quat"], seeds=(START,)), hand_q=g["pre"], duration=2.0)
    s.cart_move(pre, g["quat"], 1.0)
    s.cart_move(target, g["quat"], 1.0)
    s.move_hand(g["close"], 0.8)
    s.hold(0.3)
    s.cart_move(target + [0, 0, 0.15], g["quat"], 1.2)
    s.hold(2.0)
    rise = s.body_pos(g["obj"])[2] - z0
    touch = sorted(s.hand_contacts(g["obj"]))
    ok = rise > 0.12 and len(touch) > 0
    if place:
        s.cart_move(target + [0, 0, 0.005], g["quat"], 1.2)
        s.hold(0.5)  # let the arm settle so the object is down before release
        s.move_hand(H.OPEN, 0.6)  # thumb back into the palm plane so nothing props the object up
        s.hold(0.4)
        s.cart_move(pre, g["quat"], 0.8)
        s.cart_move(above, g["quat"], 0.8)
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
