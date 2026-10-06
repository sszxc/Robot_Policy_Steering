"""Top-down skeleton of the tracked Leap hand (core of leapc-python-bindings examples/visualiser.py)."""
import time

import cv2
import numpy as np

NAME = "Leap hand"
H, W = 500, 700
FPS = 30  # redraw rate; drawing every loop slows the teleop loop below the Leap rate
_last = 0.0
DEPTH_MM = 140  # colour range: wrist height +- this

# Keypoint pairs (26-point layout, see leap_source.py): wrist->CMC, finger chains, CMC/MCP knuckle rows.
BONES = [(0, 1 + 5 * d) for d in range(5)]
BONES += [(1 + 5 * d + k, 2 + 5 * d + k) for d in range(5) for k in range(4)]
BONES += [(1 + 5 * d, 6 + 5 * d) for d in range(4)] + [(2 + 5 * d, 7 + 5 * d) for d in range(1, 4)]


def depth_colors(kp):
    """Height above the wrist (Leap y, toward the top-down viewer) -> BGR colours (TURBO: red near, blue far)."""
    t = np.clip((kp[:, 1] - kp[0, 1]) * 1000 / (2 * DEPTH_MM) + 0.5, 0, 1)
    return cv2.applyColorMap((t * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0], t


def draw(kp, status, rec=False, dots=False):
    """kp: 26x3 Leap keypoints (m) or None. Leap x -> right, z (toward you) -> down, 1 px/mm; far drawn first."""
    img = np.zeros((H, W, 3), np.uint8)
    if kp is not None:
        px = (kp[:, [0, 2]] * 1000 + [W / 2, H / 2]).astype(int)
        col, t = depth_colors(kp)
        if not dots:
            for a, b in sorted(BONES, key=lambda e: t[e[0]] + t[e[1]]):
                cv2.line(img, tuple(px[a]), tuple(px[b]), ((col[a].astype(int) + col[b]) // 2).tolist(), 2, cv2.LINE_AA)
        for i in np.argsort(t):
            cv2.circle(img, tuple(px[i]), int(2 + 4 * t[i]), col[i].tolist(), -1, cv2.LINE_AA)
    cv2.putText(img, status, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 44), 1)
    if rec:
        cv2.circle(img, (18, 20), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "REC", (32, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img


def show(kp, status, rec=False, dots=False):
    """Draw and pump the window (at most FPS); returns False once the user closed it."""
    global _last
    if time.time() - _last < 1 / FPS:
        return True
    _last = time.time()
    cv2.imshow(NAME, draw(kp, status, rec, dots))
    cv2.waitKey(1)
    return cv2.getWindowProperty(NAME, cv2.WND_PROP_VISIBLE) >= 1


def close():
    cv2.destroyAllWindows()
