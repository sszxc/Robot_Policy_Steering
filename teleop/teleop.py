"""Live Leap -> floating HX5 hand in sim.

Keys (global, pynput): t = clutch (freeze wrist position; press again to resume from the new hand position),
r = start/stop recording raw keypoints to out/teleop/<time>.npz, q = quit.
Usage: python -m teleop.teleop [--scale 1.0] [--alpha 0.5] [--wrist leap|optitrack]
"""
import argparse
import threading
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from pynput import keyboard

from teleop import hand_sim, mapping
from teleop.leap_source import LeapSource
from teleop.wrist import leap_wrist, optitrack_wrist

OUT = Path(__file__).resolve().parents[1] / "out" / "teleop"


def save(frames):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / time.strftime("%Y%m%d_%H%M%S.npz")
    t, kp = zip(*frames)
    np.savez(path, t=np.array(t), kp=np.array(kp))
    print(f"saved {len(frames)} frames -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="wrist translation scale")
    ap.add_argument("--alpha", type=float, default=0.5, help="EMA weight of the new finger target (1 = no filter)")
    ap.add_argument("--wrist", choices=["leap", "optitrack"], default="leap")
    args = ap.parse_args()

    keys = {"t": False, "r": False, "q": False}  # pending presses
    lock = threading.Lock()

    def on_press(k):
        c = getattr(k, "char", None)
        if c in keys:
            with lock:
                keys[c] = True

    keyboard.Listener(on_press=on_press).start()

    m, d = hand_sim.build()
    src = LeapSource("right")
    q = np.zeros(20)
    offset = None  # world wrist pos -> mocap pos; None = recompute on the next frame
    tracking, recording, frames, last_id = True, False, [], None
    print(__doc__)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time.time()
        while viewer.is_running():
            with lock:
                pressed = dict(keys)
                keys.update(t=False, r=False, q=False)
            if pressed["q"]:
                break
            if pressed["t"]:
                tracking = not tracking
                offset = None
                print("wrist position", "tracking" if tracking else "frozen")
            if pressed["r"]:
                if recording and frames:
                    save(frames)
                recording, frames = not recording, []
                print("recording" if recording else "recording stopped")

            s = src.latest()
            if s is not None and s[0] != last_id:
                last_id, t, kp = s
                if recording:
                    frames.append((t, kp))
                q = args.alpha * mapping.retarget(kp) + (1 - args.alpha) * q
                hand_sim.set_fingers(d, q)
                pos, R = leap_wrist(kp) if args.wrist == "leap" else optitrack_wrist()
                if tracking:
                    if offset is None:
                        offset = d.mocap_pos[0] - args.scale * pos
                    pos = args.scale * pos + offset
                else:
                    pos = d.mocap_pos[0].copy()
                hand_sim.set_wrist(d, pos, R)

            hand_sim.step_to(m, d, time.time() - t0)
            viewer.sync()
            time.sleep(0.005)

    if recording and frames:
        save(frames)
    src.close()


if __name__ == "__main__":
    main()
