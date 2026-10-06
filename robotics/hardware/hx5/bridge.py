"""Runs inside the robotis_hand container: UDP joint targets (link.py) -> hand_r_controller trajectory topic.

Safety: clamp to the vendor limits; rate-limit from the measured pose (so the first command ramps in);
after --timeout without packets stop publishing (the controller holds its last goal) and re-ramp on resume.
Usage (container): RMW_IMPLEMENTATION=rmw_zenoh_cpp python3 /thesis/robotics/hardware/hx5/bridge.py [--max_vel 2.0]
"""
import argparse
import socket
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import link  # same directory; the script is run directly

TOPIC = "/leader/joint_trajectory_command_broadcaster_right_hand/joint_trajectory"
JOINTS = [f"finger_r_joint{i}" for i in range(1, 21)]
# Vendor limits (robotis_hand_teleop hx5_d20_teleop.py, right hand).
LO = np.array([-1.57, -1.57, -0.5, -0.5] + [-0.6, 0.0, 0.0, 0.0] * 4)
HI = np.array([1.57, 0.0, 1.57, 1.57] + [0.6, 2.0, 1.57, 1.57] * 4)
RATE = 100  # Hz, = controller_manager update_rate


class Bridge(Node):
    def __init__(self, args):
        super().__init__("hx5_bridge")
        self.args = args
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((link.HOST, args.port))
        self.sock.setblocking(False)
        self.pub = self.create_publisher(JointTrajectory, TOPIC, 10)
        self.create_subscription(JointState, "/joint_states", self.on_state, 10)
        self.meas = None  # measured q (JOINTS order)
        self.target = self.cmd = None  # cmd = last published, None = re-init from meas
        self.seq, self.last_rx = -1, 0.0
        self.create_timer(1 / RATE, self.tick)
        self.get_logger().info(f"listening on udp {link.HOST}:{args.port} -> {TOPIC}")

    def on_state(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(j in idx for j in JOINTS):
            self.meas = np.array([msg.position[idx[j]] for j in JOINTS])

    def recv(self):
        while True:
            try:
                data = self.sock.recv(link.PACKET.size)
            except BlockingIOError:
                return
            if len(data) != link.PACKET.size:
                continue
            seq, *q = link.PACKET.unpack(data)
            if seq > self.seq:  # drop out-of-order packets
                self.seq, self.target, self.last_rx = seq, np.clip(q, LO, HI), time.time()

    def tick(self):
        self.recv()
        if time.time() - self.last_rx > self.args.timeout:
            if self.cmd is not None:
                self.get_logger().info("no packets: holding")
            self.cmd, self.seq = None, -1  # seq reset lets a restarted sender through
            return
        if self.cmd is None:
            if self.meas is None:
                self.get_logger().warn("waiting for /joint_states", throttle_duration_sec=1.0)
                return
            self.cmd = self.meas.copy()
            self.get_logger().info("streaming")
        step = self.args.max_vel / RATE
        self.cmd = self.cmd + np.clip(self.target - self.cmd, -step, step)

        pt = JointTrajectoryPoint(positions=self.cmd.tolist())
        ns = int(self.args.horizon * 1e9)
        pt.time_from_start = Duration(sec=ns // 10**9, nanosec=ns % 10**9)
        self.pub.publish(JointTrajectory(joint_names=JOINTS, points=[pt]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=link.PORT)
    ap.add_argument("--max_vel", type=float, default=2.0, help="per-joint rate limit (rad/s)")
    ap.add_argument("--timeout", type=float, default=0.2, help="s without packets -> stop publishing")
    ap.add_argument("--horizon", type=float, default=0.05, help="time_from_start of each trajectory point (s)")
    args = ap.parse_args()
    rclpy.init()
    node = Bridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
