"""HX5-D20 hand control vector (20 position targets, rad) and named postures.

Order per finger: thumb = [cmc, mcp_yaw, mcp_pitch, ip]; others = [mcp_roll, mcp_pitch, pip, dip].
Ranges: thumb cmc ±1.57, yaw -1.57..0, pitch/ip 0..1.57; finger roll ±0.61, mcp 0..2.0, pip/dip 0..1.57.
Adjacent straight fingers touch after ~0.05 rad inward roll (same on the vendored mesh model); roll is for spreading.
"""
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
IDX = {f: slice(4 * i, 4 * i + 4) for i, f in enumerate(FINGERS)}


def pose(thumb=(0, 0, 0, 0), index=(0, 0, 0, 0), middle=(0, 0, 0, 0), ring=(0, 0, 0, 0), pinky=(0, 0, 0, 0)):
    return np.concatenate([thumb, index, middle, ring, pinky]).astype(float)


def blend(a, b, s):
    return (1 - s) * np.asarray(a) + s * np.asarray(b)


OPEN = pose()
CURL = (0, 1.6, 1.5, 1.5)  # finger fully curled; as a grasp target it overshoots and squeezes
# Grasp postures tuned in tasks/grasps.py (thumb/index tips meet around a ~3.5 cm object).
PINCH = pose(thumb=(0, -1.56, 0.65, 0.1), index=(0, 1.75, 0, 0.1), middle=CURL, ring=CURL, pinky=CURL)
TRIPOD = pose(thumb=(0.24, -1.57, 0.7, 0.1), index=(0, 1.75, 0, 0.1), middle=(0, 1.8, 0, 0.1), ring=CURL, pinky=CURL)
KEY_PINCH = pose(thumb=(1.15, -0.36, 0.12, 0.6), index=(0, 1.0, 1.08, 1.0), middle=CURL, ring=CURL, pinky=CURL)
HOOK = pose(index=(0, 0.3, 1.5, 1.4), middle=(0, 0.3, 1.5, 1.4), ring=(0, 0.3, 1.5, 1.4), pinky=(0, 0.3, 1.5, 1.4))
POWER = pose(thumb=(-0.5, -1.4, 1.0, 0.8), index=(0, 1.3, 1.5, 1.2), middle=(0, 1.3, 1.5, 1.2),
             ring=(0, 1.3, 1.5, 1.2), pinky=(0, 1.3, 1.5, 1.2))
