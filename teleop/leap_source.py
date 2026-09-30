"""Leap Motion -> raw keypoints (26x3, Leap frame, metres).

Layout: 0 = wrist; 1 + 5*d + k = digit d (thumb..pinky), joint k (0 CMC, 1 MCP, 2 PIP, 3 DIP, 4 tip).
Leap's thumb metacarpal has zero length, so for the thumb k=0 and k=1 coincide.
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


class LeapSource:
    """Background Leap connection; latest() returns (frame_id, t, kp) or None when the hand is not tracked."""

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
                        out = (event.tracking_frame_id, time.time(), keypoints(h))
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
