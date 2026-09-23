import numpy as np

from sim import Sim
import tasks.screwdriver as T


def test_index_gait_turns_handle_clockwise():
    s = Sim(T.build().compile())
    loops = {f: T.finger_loop(s, f) for f in T.SWEEP}
    s.reset(hand_q=T.BASE)
    s.move_hand(T.gait_pose(loops, 0.0, 1.2), 0.5)
    a0, dt = T.angle(s), s.m.opt.timestep
    for _ in range(3):
        for i in range(int(1.2 / dt)):
            s.d.ctrl[s.hand_ctrl] = T.gait_pose(loops, i * dt, 1.2)
            s.step()
    assert np.degrees(a0 - T.angle(s)) > 90  # ~38 deg/cycle expected
