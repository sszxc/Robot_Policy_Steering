"""Top-down skeleton of the tracked Leap hand (core of leapc-python-bindings examples/visualiser.py)."""
import time

import cv2
import numpy as np

NAME = "Leap hand"
H, W = 500, 700
FPS = 30  # redraw rate; drawing every loop slows the teleop loop below the Leap rate
_last = 0.0

# Keypoint pairs (26-point layout, see leap_source.py): wrist->CMC, finger chains, CMC/MCP knuckle rows.
BONES = [(0, 1 + 5 * d) for d in range(5)]
BONES += [(1 + 5 * d + k, 2 + 5 * d + k) for d in range(5) for k in range(4)]
BONES += [(1 + 5 * d, 6 + 5 * d) for d in range(4)] + [(2 + 5 * d, 7 + 5 * d) for d in range(1, 4)]


def draw(kp, status):
    """kp: 26x3 Leap keypoints (m) or None. Leap x -> right, z (toward you) -> down, 1 px/mm."""
    img = np.zeros((H, W, 3), np.uint8)
    if kp is not None:
        px = (kp[:, [0, 2]] * 1000 + [W / 2, H / 2]).astype(int)
        for a, b in BONES:
            cv2.line(img, tuple(px[a]), tuple(px[b]), (255, 255, 255), 2)
        for p in px:
            cv2.circle(img, tuple(p), 3, (255, 255, 255), -1)
    cv2.putText(img, status, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 44), 1)
    return img


def show(kp, status):
    """Draw and pump the window (at most FPS); returns False once the user closed it."""
    global _last
    if time.time() - _last < 1 / FPS:
        return True
    _last = time.time()
    cv2.imshow(NAME, draw(kp, status))
    cv2.waitKey(1)
    return cv2.getWindowProperty(NAME, cv2.WND_PROP_VISIBLE) >= 1


def close():
    cv2.destroyAllWindows()
