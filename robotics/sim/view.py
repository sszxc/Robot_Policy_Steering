"""Interactive base-scene viewer: python -m robotics.sim.view [hand|arm|lab]."""

def main():
    import argparse
    import mujoco
    from mujoco import viewer
    from robotics.sim import scene

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=("hand", "arm", "lab"), nargs="?", default="arm")
    args = parser.parse_args()
    if args.scene == "lab":
        m = scene.lab_scene().compile()
        d = mujoco.MjData(m)
        d.qpos[:6] = d.ctrl[:6] = scene.LAB_HOME
        viewer.launch(m, d)
        return
    spec = scene.arm_hand() if args.scene == "arm" else scene.hand_only()
    viewer.launch(spec.compile())


if __name__ == "__main__":
    main()
