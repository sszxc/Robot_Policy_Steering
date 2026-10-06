"""Joint-space mapping: human keypoints (26x3, see leap_source) -> 20 HX5 joint targets (sim/hand.py order).

Hand frame (human and robot alike): x = palm normal (out of the palm), y = thumb side, z = wrist -> fingers.
Each output = GAIN * geometric angle + OFFSET, then clipped to the joint range; GAIN/OFFSET presets in CALIB.
"""
import numpy as np

WRIST = 0
THUMB, INDEX, MIDDLE, RING, PINKY = range(5)

# Per joint, same order as the output (thumb cmc, yaw, pitch, ip, then roll, mcp, pip, dip per finger).
# (GAIN, OFFSET) presets, fit by calibrate.py on assets/teleop/20261006_rom_*.npz (see its docstring).
CALIB = {
    "raw": (np.ones(20), np.zeros(20)),
    "anchor": (np.array([1.000, 1.000, 2.877, 1.884, 1.706, 1.699, 1.034, 1.994, 2.039, 1.465, 0.998, 1.798, 1.779, 1.432, 1.006, 1.959, 0.878, 1.378, 1.117, 1.708]),
               np.array([0.000, 0.000, 0.000, 0.000, -0.071, 0.000, 0.000, 0.000, -0.303, 0.000, 0.000, 0.000, -0.148, 0.000, 0.000, 0.000, -0.310, 0.000, 0.000, 0.000])),
    "full": (np.array([2.527, 1.478, 1.854, 1.775, 2.009, 1.178, 1.044, 2.106, 2.257, 1.120, 1.017, 1.888, 1.957, 1.160, 1.055, 2.072, 1.360, 1.056, 1.173, 1.862]),
             np.array([-3.110, -0.167, 0.559, 0.091, -0.192, 0.616, -0.014, -0.088, -0.270, 0.473, -0.031, -0.079, -0.101, 0.382, -0.078, -0.090, -0.145, 0.469, -0.078, -0.142])),
}
DEFAULT_CALIB = "anchor"
LO = np.array([-1.571, -1.571, 0, 0] + [-0.611, 0, 0, 0] * 4)
HI = np.array([1.571, 0, 1.571, 1.571] + [0.611, 2.007, 1.571, 1.571] * 4)


def joint(kp, digit, k):
    return kp[1 + 5 * digit + k]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def hand_frame(kp):
    """3x3 rotation, columns = hand x/y/z axes, in the keypoints' coordinates (right hand)."""
    z = unit(joint(kp, MIDDLE, 1) - kp[WRIST])
    y = joint(kp, INDEX, 1) - joint(kp, PINKY, 1)
    y = unit(y - (y @ z) * z)
    return np.column_stack([np.cross(y, z), y, z])


def flex(a, b, axis):
    """Signed angle from segment a to b about axis (right-handed)."""
    a, b = unit(a), unit(b)
    return np.arctan2(np.cross(a, b) @ axis, a @ b)


def angles(kp):
    """26x3 keypoints (any frame) -> 20 raw geometric joint angles (rad), before GAIN/OFFSET/clip."""
    kp = kp @ hand_frame(kp)  # express everything in the hand frame
    q = np.zeros(20)

    # Thumb: robot metacarpal dir = Rx(cmc) Rz(yaw) y = (-sin yaw, cos yaw cos cmc, cos yaw sin cmc).
    m = unit(joint(kp, THUMB, 2) - joint(kp, THUMB, 1))
    p = joint(kp, THUMB, 3) - joint(kp, THUMB, 2)
    d = joint(kp, THUMB, 4) - joint(kp, THUMB, 3)
    yaw = -np.arcsin(np.clip(m[0], -1, 1))
    # (m1, m2) = cos(yaw)(cos cmc, sin cmc) is singular at yaw=-90 (pinch); the MCP flexion plane gives
    # link2 z = (0, -sin cmc, cos cmc) scaled by sin(flex). Summing weights each by how well-conditioned it is.
    z2 = np.cross(np.cross(m, unit(p)), m)
    cmc = np.arctan2(m[2] - z2[1], m[1] + z2[2])
    axis = np.array([np.cos(yaw), np.sin(yaw) * np.cos(cmc), np.sin(yaw) * np.sin(cmc)])  # Rx(cmc) Rz(yaw) x
    q[0:4] = cmc, yaw, flex(m, p, axis), flex(p, d, axis)

    # Fingers: proximal dir in the metacarpal frame; roll = asin fades out as the finger curls.
    for f in (INDEX, MIDDLE, RING, PINKY):
        s = [joint(kp, f, k + 1) - joint(kp, f, k) for k in range(4)]  # metacarpal, proximal, middle, distal
        z = unit(s[0])
        x = unit(np.array([1.0, 0, 0]) - z[0] * z)
        y = np.cross(z, x)
        p = unit(s[1]) @ np.column_stack([x, y, z])
        roll = np.arcsin(np.clip(-p[1], -1, 1))
        pitch = np.arctan2(p[0], p[2])
        q[4 * f:4 * f + 4] = roll, pitch, flex(s[1], s[2], y), flex(s[2], s[3], y)

    return q


def retarget(kp, calib=DEFAULT_CALIB):
    """26x3 keypoints (any frame) -> 20 joint targets (rad), with the CALIB[calib] gain/offset."""
    gain, offset = CALIB[calib]
    return np.clip(gain * angles(kp) + offset, LO, HI)
