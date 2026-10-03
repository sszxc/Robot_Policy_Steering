"""Streaming servoJ/servoL: one sine period on a joint, then a TCP circle. Keep speed slider low."""

def main():
    import argparse
    import os
    from platforms.paths import platform_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default=os.environ.get("THESIS_UR5_IP", platform_config().get("ur5_ip", "192.168.1.132")))
    args = parser.parse_args()

    import math
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    IP = args.ip
    c, r = RTDEControlInterface(IP), RTDEReceiveInterface(IP)

    DT, T, A, R = 1 / 125, 4.0, 0.3, 0.03   # CB3 control rate 125 Hz; period s; joint amp rad; radius m
    STEPS = int(T / DT)
    q0, p0 = r.getActualQ(), r.getActualTCPPose()

    # servoJ args: q, speed, accel (unused), time, lookahead 0.03-0.2, gain 100-2000
    for i in range(STEPS + 1):
        t0 = c.initPeriod()
        q = list(q0); q[5] += A * math.sin(2 * math.pi * i / STEPS)
        c.servoJ(q, 0, 0, DT, 0.1, 300)
        c.waitPeriod(t0)
    c.servoStop()

    # # servoL args: pose, speed, accel (both unused), time, lookahead, gain
    # for i in range(STEPS + 1):
    #     t0 = c.initPeriod()
    #     a = 2 * math.pi * i / STEPS   # circle in base xy plane, passes through p0
    #     p = list(p0); p[0] += R * (math.cos(a) - 1); p[1] += R * math.sin(a)
    #     c.servoL(p, 0, 0, DT, 0.1, 300)
    #     c.waitPeriod(t0)
    # c.servoStop()

    c.stopScript()


if __name__ == "__main__":
    main()
