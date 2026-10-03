import numpy as np

from robotics.policies.scripted import baoding as B


def test_balls_rest_in_hand_and_gait_in_range():
    p = (B.LOW + B.HIGH) / 2
    ang, dropped, _ = B.rollout(p, duration=1.0)
    assert not dropped and np.isfinite(ang)
    lo, hi = B.HAND_RANGE
    q = B.gait(p, 0.3)
    assert np.all(q >= lo) and np.all(q <= hi)
