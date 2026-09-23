"""Interactive viewer for the base scenes: python view.py [hand|arm]"""
import sys

from mujoco import viewer

from sim import scene

spec = scene.arm_hand() if (sys.argv[1:] or ["arm"])[0] == "arm" else scene.hand_only()
m = spec.compile()
viewer.launch(m)
