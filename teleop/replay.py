"""Replay a recording (out/teleop/*.npz) through the current mapping.py in sim.

Keys (global, pynput): space = pause/resume, q = quit.
Usage: python -m teleop.replay out/teleop/<file>.npz [--speed 1.0] [--loop] [--fix_wrist]
"""
import argparse
import threading
import time

import mujoco.viewer
import numpy as np
from pynput import keyboard

from teleop import hand_sim, mapping
from teleop.wrist import leap_wrist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--fix_wrist", action="store_true", help="keep the wrist at home, fingers only")
    args = ap.parse_args()

    rec = np.load(args.path)
    t, kp = rec["t"] - rec["t"][0], rec["kp"]
    print(f"{len(t)} frames, {t[-1]:.1f} s")

    state = {"pause": False, "quit": False}

    def on_press(k):
        if k == keyboard.Key.space:
            state["pause"] = not state["pause"]
        elif getattr(k, "char", None) == "q":
            state["quit"] = True

    keyboard.Listener(on_press=on_press).start()

    m, d = hand_sim.build()
    offset = hand_sim.HOME_POS - leap_wrist(kp[0])[0]
    with mujoco.viewer.launch_passive(m, d) as viewer:
        i, clock, sim_t, last = 0, 0.0, 0.0, time.time()
        while viewer.is_running() and not state["quit"]:
            now = time.time()
            dt = 0.0 if state["pause"] else (now - last) * args.speed
            clock, sim_t, last = clock + dt, sim_t + dt, now
            while i < len(t) and t[i] <= clock:
                hand_sim.set_fingers(d, mapping.retarget(kp[i]))
                if not args.fix_wrist:
                    pos, R = leap_wrist(kp[i])
                    hand_sim.set_wrist(d, pos + offset, R)
                i += 1
            if i == len(t) and args.loop:
                i, clock = 0, 0.0
            hand_sim.step_to(m, d, sim_t)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
