import sys

from robotics.policies.scripted import scissors


def test_scissors_open_close():
    sys.argv = ["scissors"]
    scissors.CYCLES = 3
    assert scissors.main()
