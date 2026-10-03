"""Interactive base-scene viewer: python -m robotics.sim.view [hand|arm]."""

def main():
    import argparse
    from mujoco import viewer
    from robotics.sim import scene

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=("hand", "arm"), nargs="?", default="arm")
    args = parser.parse_args()
    spec = scene.arm_hand() if args.scene == "arm" else scene.hand_only()
    viewer.launch(spec.compile())


if __name__ == "__main__":
    main()
