"""Wrist pose sources -> (pos, R) in the sim world frame; R columns = hand x/y/z (see mapping.py)."""
import numpy as np

from robotics.teleop.mapping import hand_frame

# Leap (x right, y up, z toward the operator) -> world (x away from the operator, y left, z up).
LEAP2WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)


def leap_wrist(kp):
    return LEAP2WORLD @ kp[0], LEAP2WORLD @ hand_frame(kp)


def optitrack_wrist():
    raise NotImplementedError("OptiTrack wrist source not implemented yet")
