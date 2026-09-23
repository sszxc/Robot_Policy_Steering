"""Compose MuJoCo scenes (hand-only or UR5e+hand, plus objects) with MjSpec."""
from pathlib import Path

import mujoco
import numpy as np

MJ = Path(__file__).resolve().parents[1] / "mujoco"

# UR5e joints: palm site at (0.5, 0, 0.6), palm facing down, fingers along +x (table in front at x>0).
ARM_HOME = np.array([2.723, -1.776, 1.917, -3.283, -1.153, 1.571])


def hand_quat(palm_dir, finger_dir):
    """Quaternion placing the hand so its palm normal / finger axis point along given world dirs."""
    x = np.asarray(palm_dir, float); x /= np.linalg.norm(x)
    z = np.asarray(finger_dir, float); z -= x * (x @ z); z /= np.linalg.norm(z)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, np.cross(z, x), z]).ravel())
    return q


def _base():
    return mujoco.MjSpec.from_file(str(MJ / "scene_base.xml"))


def _hand():
    return mujoco.MjSpec.from_file(str(MJ / "hx5_right.xml"))


def hand_only(pos=(0, 0, 0.3), quat=None):
    """Hand base welded to the world at pos/quat (default: palm up, fingers along +x)."""
    spec = _base()
    quat = hand_quat((0, 0, 1), (1, 0, 0)) if quat is None else quat
    frame = spec.worldbody.add_frame(pos=pos, quat=quat)
    frame.attach_body(_hand().body("hand_base"), "", "")
    return spec


def arm_hand():
    """UR5e at the world origin with the hand bolted to its flange (no adapter yet)."""
    spec = _base()
    arm = mujoco.MjSpec.from_file(str(MJ / "universal_robots_ur5e" / "ur5e.xml"))
    for k in list(arm.keys):
        arm.delete(k)
    arm.site("attachment_site").attach_body(_hand().body("hand_base"), "", "")
    spec.worldbody.add_frame().attach_body(arm.body("base"), "", "")
    return spec


def add_table(spec, pos=(0.55, 0, 0), size=(0.3, 0.4), height=0.4):
    """Box table whose top surface is at z=height."""
    b = spec.worldbody.add_body(name="table", pos=[pos[0], pos[1], height / 2])
    b.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[size[0], size[1], height / 2],
               material="wood", condim=3, friction=[1, 0.01, 0.001])
    return height


def add_object(spec, xml, pos, quat=(1, 0, 0, 0), prefix=""):
    """Attach the first top-level body of an MJCF file (e.g. mujoco/objects/cube.xml) at pos/quat."""
    child = mujoco.MjSpec.from_file(str(MJ / "objects" / xml) if "/" not in str(xml) else str(xml))
    frame = spec.worldbody.add_frame(pos=list(pos), quat=list(quat))
    return frame.attach_body(child.worldbody.first_body(), prefix, "")
