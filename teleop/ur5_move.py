"""Small joint move and TCP move, each returning to start. Keep speed slider low."""
import math
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

IP = "192.168.1.132"
c, r = RTDEControlInterface(IP), RTDEReceiveInterface(IP)

# # Joint move: wrist 3 (last joint) +0.1 rad, then back. args: target, speed rad/s, accel rad/s^2
q0 = r.getActualQ()
# q1 = list(q0); q1[6] += 0.1
# c.moveJ(q1, 0.2, 0.5)
# print("q after moveJ:", r.getActualQ())
# c.moveJ(q0, 0.2, 0.5)

# # TCP move: +z 3 cm in base frame, then back. args: pose, speed m/s, accel m/s^2
p0 = r.getActualTCPPose()
# p1 = list(p0); p1[2] += 0.03
# c.moveL(p1, 0.05, 0.2)
# print("tcp after moveL:", r.getActualTCPPose())
# c.moveL(p0, 0.05, 0.2)

# # # Waypoint loop with blocking moveJ/moveL (stops at each point; expect stutter)
N, T, A, R = 40, 4.0, 0.3, 0.03   # points, period s, joint amp rad, circle radius m
# # for i in range(N + 1):
# #     q = list(q0); q[5] += A * math.sin(2 * math.pi * i / N)
# #     c.moveJ(q, 0.5, 2.0)
# # c.moveJ(q0, 0.2, 0.5)
for i in range(N + 1):
    a = 2 * math.pi * i / N   # circle in base xy plane, passes through p0
    p = list(p0); p[0] += R * (math.cos(a) - 1); p[1] += R * math.sin(a)
    c.moveL(p, 0.1, 0.5)
c.moveL(p0, 0.05, 0.2)

c.stopScript()
