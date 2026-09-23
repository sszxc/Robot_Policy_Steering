"""HX5-D20 hand control vector (20 position targets, rad) and named postures.

Order per finger: thumb = [cmc, mcp_yaw, mcp_pitch, ip]; others = [mcp_roll, mcp_pitch, pip, dip].
Ranges: thumb cmc ±1.57, yaw -1.57..0, pitch/ip 0..1.57; finger roll ±0.61, mcp 0..2.0, pip/dip 0..1.57.
"""
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
IDX = {f: slice(4 * i, 4 * i + 4) for i, f in enumerate(FINGERS)}


def pose(thumb=(0, 0, 0, 0), index=(0, 0, 0, 0), middle=(0, 0, 0, 0), ring=(0, 0, 0, 0), pinky=(0, 0, 0, 0)):
    return np.concatenate([thumb, index, middle, ring, pinky]).astype(float)


def blend(a, b, s):
    return (1 - s) * np.asarray(a) + s * np.asarray(b)


OPEN = pose()
