"""Rerun view of baoding_server rollouts: raw wrist frame (live only, JPEG), policy input frame, finger state vs
target, action, image-head ball estimate, timing. Live: baoding_server --rerun [--save].
Saved rollouts: python -m robotics.policies.rl.baoding_rerun out/rl/baoding_real/server/<time>.npz ... [--save]"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from robotics.policies.rl.baoding_env import CENTER

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
COLORS = np.array([[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40]])


def start(student, name, rrd=None, view=True, raw=False):
    """New recording (one per rollout) -> the viewer and / or an .rrd file. raw: a raw-camera tab first."""
    rr.init(f"baoding_{student}")
    rr.send_recording_name(name)
    if view:
        rr.spawn(connect=False)  # no-op when a viewer is already open
    rr.set_sinks(*([rr.GrpcSink()] if view else []), *([rr.FileSink(str(rrd))] if rrd else []))
    plots = [rrb.TimeSeriesView(origin=f, name=f) for f in FINGERS]
    plots += [rrb.TimeSeriesView(origin=f"action/{f}", name=f"{f} action") for f in FINGERS]
    cams = [rrb.Spatial2DView(origin="camera/wrist", name="wrist (policy input)")]
    if raw:
        cams.insert(0, rrb.Spatial2DView(origin="camera/raw", name="wrist raw"))
    left = rrb.Vertical(rrb.Tabs(*cams),
                        *[rrb.TimeSeriesView(origin=o, name=n) for o, n in
                          [("balls", "image head: ball y, z (mm from orbit center)"), ("phase", "pair angle (deg)"),
                           ("timing", "timing (ms)")]], row_shares=[3, 2, 2, 1])
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(left, rrb.Grid(*plots, grid_columns=5), column_shares=[1, 3]),
                                    collapse_panels=True))
    for f in FINGERS:  # state thick, target thin and lighter, same color per joint
        rr.log(f"{f}/state", rr.SeriesLines(colors=COLORS, widths=2.0, names=[f"j{k} state" for k in range(1, 5)]), static=True)
        rr.log(f"{f}/target", rr.SeriesLines(colors=(COLORS + 255) // 2, widths=1.0,
                                             names=[f"j{k} target" for k in range(1, 5)]), static=True)
        rr.log(f"action/{f}", rr.SeriesLines(colors=COLORS, names=[f"j{k}" for k in range(1, 5)]), static=True)
    for ball, color in [("white", [150, 150, 150]), ("yellow", [219, 200, 48])]:
        rr.log(f"balls/{ball}", rr.SeriesLines(colors=[color, color], widths=[2.0, 1.0], names=[f"{ball} y", f"{ball} z"]), static=True)
    rr.log("phase/seen", rr.SeriesLines(colors=COLORS[:1], names=["image head"]), static=True)
    rr.log("phase/goal", rr.SeriesLines(colors=COLORS[1:2], names=["goal"]), static=True)
    rr.log("timing", rr.SeriesLines(colors=COLORS[:2], names=["image age", "inference"]), static=True)


def log(i, frame, vis_state, action, target, balls, theta, stamps, infer_ms, raw=None):
    """One step, the fields of a baoding_server log record; raw = the camera RGB frame as received (~1.6 ms JPEG)."""
    rr.set_time("step", sequence=i)
    rr.set_time("stamp", timestamp=float(stamps[2]))
    if raw is not None:
        jpg = cv2.imencode(".jpg", np.ascontiguousarray(raw[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
        rr.log("camera/raw", rr.EncodedImage(contents=jpg.tobytes(), media_type="image/jpeg"))
    rr.log("camera/wrist", rr.Image(frame))
    for k, f in enumerate(FINGERS):
        s = slice(4 * k, 4 * k + 4)
        rr.log(f"{f}/state", rr.Scalars(vis_state[s]))
        rr.log(f"{f}/target", rr.Scalars(target[s]))
        rr.log(f"action/{f}", rr.Scalars(action[s]))
    b = 1e3 * (balls - CENTER)[:, 1:]
    rr.log("balls/white", rr.Scalars(b[0]))
    rr.log("balls/yellow", rr.Scalars(b[1]))
    rr.log("phase/seen", rr.Scalars(np.degrees(np.arctan2(b[0, 1] - b[1, 1], b[0, 0] - b[1, 0]))))
    if np.any(vis_state[60:66]):  # tracking students only
        rr.log("phase/goal", rr.Scalars(np.degrees((theta + np.pi) % (2 * np.pi) - np.pi)))
    rr.log("timing", rr.Scalars([1e3 * (stamps[2] - stamps[1]), infer_ms]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+")
    p.add_argument("--save", action="store_true", help="write <npz stem>.rrd next to each file instead of opening the viewer")
    a = p.parse_args()
    keys = ("frame", "vis_state", "action", "target", "balls", "theta", "stamps", "infer_ms")
    for path in map(Path, a.npz):
        d = np.load(path)
        student = str(d["student"]) if "student" in d else "rollout"
        start(student, path.stem, path.with_suffix(".rrd") if a.save else None, view=not a.save)
        for i, rec in enumerate(zip(*(d[k] for k in keys))):
            log(i, *rec)
