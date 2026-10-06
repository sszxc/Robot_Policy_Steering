"""Live Leap -> floating HX5 hand in sim, plus a top-down Leap skeleton window (leap_view).

Keys (global, pynput): t = clutch (freeze wrist position; press again to resume from the new hand position),
r = start/stop recording raw keypoints + all Leap hand fields (leap_source.hand_info) to out/teleop/<time>.npz, f = Leap view skeleton/dots,
e = (--real) start/stop sending finger targets to the real hand (off at start; needs hardware/hx5/bridge.py), q = quit. Closing either window also quits.
Usage: python -m robotics.teleop.teleop [--scale 1.0] [--alpha 0.5] [--wrist leap|optitrack] [--calib anchor|full|raw] [--real]
"""
import argparse
import threading
import time
from platforms.paths import output_path

import mujoco.viewer
import numpy as np
from pynput import keyboard

from robotics.hardware.hx5.link import HandLink
from robotics.teleop import hand_sim, leap_view, mapping
from robotics.teleop.leap_source import LeapSource
from robotics.teleop.wrist import leap_wrist, optitrack_wrist

OUT = output_path("teleop")


def save(frames):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / time.strftime("%Y%m%d_%H%M%S.npz")
    t, kp, info = zip(*frames)
    np.savez(path, t=np.array(t), kp=np.array(kp), **{k: np.array([i[k] for i in info]) for k in info[0]})
    print(f"saved {len(frames)} frames -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="wrist translation scale")
    ap.add_argument("--alpha", type=float, default=0.5, help="EMA weight of the new finger target (1 = no filter)")
    ap.add_argument("--wrist", choices=["leap", "optitrack"], default="leap")
    ap.add_argument("--calib", choices=list(mapping.CALIB), default=mapping.DEFAULT_CALIB, help="mapping gain/offset preset")
    ap.add_argument("--real", action="store_true", help="also drive the real hand (toggle with e)")
    args = ap.parse_args()

    keys = {"t": False, "r": False, "e": False, "f": False, "q": False}  # pending presses
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
    tracking, recording, frames, last_id, dots = True, False, [], None, False
    hand, live = (HandLink() if args.real else None), False
    print(__doc__)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        hand_sim.set_camera(viewer.cam)
        t0 = time.time()
        while viewer.is_running():
            with lock:
                pressed = dict(keys)
                keys.update(t=False, r=False, e=False, f=False, q=False)
            if pressed["q"]:
                break
            if pressed["t"]:
                tracking = not tracking
                offset = None
                print("wrist position", "tracking" if tracking else "frozen")
            if pressed["e"] and hand:
                live = not live
                print("real hand", "live" if live else "off")
            if pressed["f"]:
                dots = not dots
            if pressed["r"]:
                if recording and frames:
                    save(frames)
                recording, frames = not recording, []
                print("recording" if recording else "recording stopped")

            s = src.latest()
            if s is None:
                offset = None  # hand lost: re-anchor when it returns instead of jumping
            elif s[0] != last_id:
                last_id, t, kp, info = s
                if recording:
                    frames.append((t, kp, info))
                q = args.alpha * mapping.retarget(kp, args.calib) + (1 - args.alpha) * q
                hand_sim.set_fingers(d, q)
                if live:
                    hand.send(q)
                pos, R = leap_wrist(kp) if args.wrist == "leap" else optitrack_wrist()
                if tracking:
                    if offset is None:
                        offset = d.mocap_pos[0] - args.scale * pos
                    pos = args.scale * pos + offset
                else:
                    pos = d.mocap_pos[0].copy()
                hand_sim.set_wrist(d, pos, R)

            status = f"wrist {'tracking' if tracking else 'frozen'}{'' if s else '  no right hand'}"
            if hand:
                status += f"  real {'LIVE' if live else 'off'}"
            if not leap_view.show(None if s is None else s[2], status, recording, dots):
                break

            hand_sim.step_to(m, d, time.time() - t0)
            viewer.sync()
            time.sleep(0.005)

    hand_sim.close_viewer(viewer)
    if recording and frames:
        save(frames)
    leap_view.close()
    src.close()
    if hand:
        hand.close()


if __name__ == "__main__":
    main()
