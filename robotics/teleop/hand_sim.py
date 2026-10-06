"""Floating HX5 hand: free joint welded to a mocap body, plus a table and a cube."""
import threading

import mujoco
import numpy as np

from robotics.sim import scene

HOME_POS = np.array([0.45, 0.0, 0.65])
HOME_QUAT = scene.hand_quat((0, 0, -1), (1, 0, 0))  # palm down, fingers +x


def build():
    spec = scene._base()
    mocap = spec.worldbody.add_body(name="hand_mocap", pos=HOME_POS, quat=HOME_QUAT, mocap=True)
    frame = spec.worldbody.add_frame(pos=HOME_POS, quat=HOME_QUAT)
    base = frame.attach_body(scene._hand().body("hand_base"), "", "")
    base.add_freejoint()
    spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1=mocap.name, name2="hand_base", solref=[0.01, 1],
                      data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])  # anchor, relpos, relquat, torquescale
    top = scene.add_table(spec)
    scene.add_object(spec, "cube.xml", (0.55, 0, top + 0.03))
    m = spec.compile()
    return m, mujoco.MjData(m)


def set_camera(cam):
    """Operator view: behind the hand, looking along +x (away from the operator)."""
    cam.lookat[:] = (0.5, 0, 0.55)
    cam.distance, cam.azimuth, cam.elevation = 0.8, 0, -30


def close_viewer(viewer):
    """Close a passive viewer and wait for its render thread; exiting while it tears down segfaults in glfw."""
    viewer.close()
    for th in threading.enumerate():
        if "_launch_internal" in th.name:
            th.join(timeout=5)


def set_wrist(d, pos, R):
    d.mocap_pos[0] = pos
    mujoco.mju_mat2Quat(d.mocap_quat[0], np.asarray(R, float).ravel())


def set_fingers(d, q):
    d.ctrl[:20] = q


def step_to(m, d, t):
    while d.time < t:
        mujoco.mj_step(m, d)
