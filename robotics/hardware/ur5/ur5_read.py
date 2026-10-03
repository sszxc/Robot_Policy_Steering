"""Read-only UR5 state via ur_rtde. Never imports RTDEControlInterface (no motion)."""

def main():
    import argparse
    import os
    from platforms.paths import platform_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default=os.environ.get("THESIS_UR5_IP", platform_config().get("ur5_ip", "192.168.1.132")))
    args = parser.parse_args()

    from rtde_receive import RTDEReceiveInterface
    import numpy as np
    import time

    IP = args.ip

    r = RTDEReceiveInterface(IP)
    np.set_printoptions(precision=4, suppress=True)
    print("q (rad):      ", np.array(r.getActualQ()))
    print("qd (rad/s):   ", np.array(r.getActualQd()))
    print("TCP pose:     ", np.array(r.getActualTCPPose()))
    print("TCP force:    ", np.array(r.getActualTCPForce()))
    print("robot mode:   ", r.getRobotMode())
    print("safety mode:  ", r.getSafetyMode())
    print("runtime state:", r.getRuntimeState())
    print("joint temps:  ", np.array(r.getJointTemperatures()))

    for i in range(1000):
        print(f"{i:02d} {time.time():.3f}  q (rad):", np.array(r.getActualQ()))

    r.disconnect()


if __name__ == "__main__":
    main()
