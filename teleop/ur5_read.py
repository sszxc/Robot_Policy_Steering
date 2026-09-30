"""Read-only UR5 state via ur_rtde. Never imports RTDEControlInterface (no motion)."""
from rtde_receive import RTDEReceiveInterface
import numpy as np
import time

IP = "192.168.1.132"

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
