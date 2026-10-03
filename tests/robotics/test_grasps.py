"""Fast checks that the scripted grasps still lift their objects. Run: python -m pytest tests -q"""
import pytest

from robotics.sim import Sim, hand as H
from robotics.policies.scripted import grasps as G


@pytest.mark.parametrize("name", ["power", "pinch"])
def test_grasp_lifts(name):
    s = Sim(G.build([name]))
    s.reset(arm_q=G.START, hand_q=H.OPEN)
    ok, rise, touch = G.run_grasp(s, name, place=False)
    assert ok, (rise, touch)
