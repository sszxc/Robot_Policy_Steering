"""Joint-space mapping: human keypoints (26x3, see leap_source) -> 20 HX5 joint targets (sim/hand.py order).

Hand frame (human and robot alike): x = palm normal (out of the palm), y = thumb side, z = wrist -> fingers.
Each output = GAIN * geometric angle + OFFSET, then clipped to the joint range; tune GAIN/OFFSET here.
"""
import numpy as np

WRIST = 0
THUMB, INDEX, MIDDLE, RING, PINKY = range(5)

# Per joint, same order as the output (thumb cmc, yaw, pitch, ip, then roll, mcp, pip, dip per finger).
GAIN = np.ones(20)
OFFSET = np.zeros(20)
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


def retarget(kp):
    """26x3 keypoints (any frame) -> 20 joint targets (rad)."""
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

    return np.clip(GAIN * q + OFFSET, LO, HI)
