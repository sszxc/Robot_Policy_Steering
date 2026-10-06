"""Leap Motion -> raw keypoints (26x3, Leap frame, metres).

Layout: 0 = wrist; 1 + 5*d + k = digit d (thumb..pinky), joint k (0 CMC, 1 MCP, 2 PIP, 3 DIP, 4 tip).
Leap's thumb metacarpal has zero length, so for the thumb k=0 and k=1 coincide.
hand_info() keeps every other LEAP_HAND field in Leap units (mm, mm/s, us; quaternions xyzw).
"""
import threading
import time

import leap
import numpy as np


def keypoints(hand):
    pts = [hand.arm.next_joint]
    for d in hand.digits:
        b = d.bones
        pts += [b[0].prev_joint, b[1].prev_joint, b[2].prev_joint, b[3].prev_joint, b[3].next_joint]
    return np.array([(p.x, p.y, p.z) for p in pts]) / 1000.0


def hand_info(hand, event):
    v = lambda p: (p.x, p.y, p.z)
    q = lambda r: (r.x, r.y, r.z, r.w)
    bones = [b for d in hand.digits for b in d.bones]  # (digit, metacarpal..distal) flattened
    p = hand.palm
    return dict(
        frame_id=event.tracking_frame_id, leap_ts=event.timestamp, framerate=event.framerate,
        hand_id=hand.id, flags=hand.flags, confidence=hand.confidence, visible_time=hand.visible_time,
        pinch_distance=hand.pinch_distance, pinch_strength=hand.pinch_strength,
        grab_angle=hand.grab_angle, grab_strength=hand.grab_strength,
        palm_pos=v(p.position), palm_stab_pos=v(p.stabilized_position), palm_vel=v(p.velocity),
        palm_normal=v(p.normal), palm_dir=v(p.direction), palm_quat=q(p.orientation), palm_width=p.width,
        bone_prev=np.reshape([v(b.prev_joint) for b in bones], (5, 4, 3)),
        bone_next=np.reshape([v(b.next_joint) for b in bones], (5, 4, 3)),
        bone_quat=np.reshape([q(b.rotation) for b in bones], (5, 4, 4)),
        bone_width=np.reshape([b.width for b in bones], (5, 4)),
        finger_id=[d.finger_id for d in hand.digits], extended=[d.is_extended for d in hand.digits],
        arm_prev=v(hand.arm.prev_joint), arm_next=v(hand.arm.next_joint),
        arm_quat=q(hand.arm.rotation), arm_width=hand.arm.width,
    )


class LeapSource:
    """Background Leap connection; latest() returns (frame_id, t, kp, info) or None when the hand is not tracked."""

    def __init__(self, hand="right"):
        want = leap.HandType.Right if hand == "right" else leap.HandType.Left
        self._lock = threading.Lock()
        self._latest = None
        src = self

        class _Listener(leap.Listener):
            def on_connection_event(self, event):
                print("Leap: connected")

            def on_tracking_event(self, event):
                out = None
                for h in event.hands:
                    if h.type == want:
                        out = (event.tracking_frame_id, time.time(), keypoints(h), hand_info(h, event))
                        break
                with src._lock:
                    src._latest = out

        self._conn = leap.Connection()
        self._conn.add_listener(_Listener())
        self._ctx = self._conn.open()
        self._ctx.__enter__()
        self._conn.set_tracking_mode(leap.TrackingMode.Desktop)

    def latest(self):
        with self._lock:
            return self._latest

    def close(self):
        self._ctx.__exit__(None, None, None)
