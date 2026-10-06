"""Sweep one joint at a time on the sim hand and the real hand together, to check zero pose and signs by eye.

Each joint goes 0 -> +-amp -> 0 (towards its open range) over --period s, others stay at 0; holds 0 for 2 s first.
Needs bridge.py running in the container. Close the viewer or Ctrl-C to stop (the bridge then holds).
Usage: python -m robotics.hardware.hx5.sweep [--joints 1 6 7] [--amp 0.3] [--period 3]
"""
import argparse
import time

import mujoco.viewer
import numpy as np

from robotics.hardware.hx5.link import HandLink
from robotics.teleop import hand_sim, mapping

HOLD = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", type=int, nargs="*", default=list(range(1, 21)), help="1-based finger_r_joint numbers")
    ap.add_argument("--amp", type=float, default=0.3, help="rad")
    ap.add_argument("--period", type=float, default=3.0, help="s per joint")
    args = ap.parse_args()

    m, d = hand_sim.build()
    hand = HandLink()
    sign = np.where(mapping.HI > 0, 1.0, -1.0)
    plan = [(None, HOLD)] + [(j - 1, args.period) for j in args.joints]

    with mujoco.viewer.launch_passive(m, d) as viewer:
        hand_sim.set_camera(viewer.cam)
        t0 = time.time()
        try:
            for j, dur in plan:
                if j is not None:
                    print(f"joint {j + 1:2d}  {m.actuator(j).name}  -> {sign[j] * args.amp:+.2f} rad")
                start = time.time()
                while viewer.is_running() and (s := time.time() - start) < dur:
                    q = np.zeros(20)
                    if j is not None:
                        q[j] = sign[j] * args.amp * np.sin(np.pi * s / dur) ** 2
                    q = np.clip(q, mapping.LO, mapping.HI)
                    hand.send(q)
                    hand_sim.set_fingers(d, q)
                    hand_sim.step_to(m, d, time.time() - t0)
                    viewer.sync()
                    time.sleep(0.01)
        except KeyboardInterrupt:
            pass
    hand_sim.close_viewer(viewer)
    hand.close()


if __name__ == "__main__":
    main()
