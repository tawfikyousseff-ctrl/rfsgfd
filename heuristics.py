"""Classical OpenCV scene cues (no learning): corridor vanishing point, stair-like
repeated horizontal edges, and blank wall detection. These are HEURISTICS and will
produce false positives/negatives; they complement the YOLO detections."""
import math

import cv2
import numpy as np


def _angle(l):
    a = math.degrees(math.atan2(l[3] - l[1], l[2] - l[0]))
    if a > 90:
        a -= 180
    elif a <= -90:
        a += 180
    return a


def _eq(l):
    x1, y1, x2, y2 = l
    return y1 - y2, x2 - x1, x1 * y2 - x2 * y1  # ax + by + c = 0


def _vanishing_point(lines, w, h):
    left, right = [], []
    for l in lines:
        a = _angle(l)
        length = math.hypot(l[2] - l[0], l[3] - l[1])
        mx = (l[0] + l[2]) / 2
        if 12 < abs(a) < 75:
            if a < 0 and mx < 0.65 * w:      # "/" lines on the left side
                left.append((length, l))
            elif a > 0 and mx > 0.35 * w:    # "\" lines on the right side
                right.append((length, l))
    if len(left) < 2 or len(right) < 2:
        return None, 0.0
    left = sorted(left, key=lambda t: -t[0])[:12]
    right = sorted(right, key=lambda t: -t[0])[:12]
    pts = []
    for _, l1 in left:
        a1, b1, c1 = _eq(l1)
        for _, l2 in right:
            a2, b2, c2 = _eq(l2)
            d = a1 * b2 - a2 * b1
            if abs(d) < 1e-6:
                continue
            pts.append(((b1 * c2 - b2 * c1) / d, (c1 * a2 - c2 * a1) / d))
    if not pts:
        return None, 0.0
    pts = np.array(pts)
    mx, my = np.median(pts, axis=0)
    spread = float(np.median(np.abs(pts[:, 0] - mx)))
    if not (-0.3 * w < mx < 1.3 * w and -0.2 * h < my < 0.85 * h):
        return None, 0.0
    conf = min(1.0, (len(left) + len(right)) / 14.0) * (1.0 if spread < 0.25 * w else 0.5)
    return (float(np.clip(mx / w, 0, 1)), float(np.clip(my / h, 0, 1))), round(conf, 2)


def _stairs(lines, w, h):
    hor = []
    for l in lines:
        ym, xm = (l[1] + l[3]) / 2, (l[0] + l[2]) / 2
        length = math.hypot(l[2] - l[0], l[3] - l[1])
        if abs(_angle(l)) < 8 and ym > 0.4 * h and 0.15 * w < xm < 0.85 * w and length > 0.18 * w:
            hor.append([ym, min(l[0], l[2]), max(l[0], l[2])])
    if len(hor) < 6:
        return None
    hor.sort()
    merged = []
    for y, x1, x2 in hor:
        if merged and y - merged[-1][0] < 5:
            merged[-1][1] = min(merged[-1][1], x1)
            merged[-1][2] = max(merged[-1][2], x2)
        else:
            merged.append([y, x1, x2])
    if len(merged) < 5:
        return None
    gaps = np.diff([m[0] for m in merged])
    growing = float(np.mean(gaps[1:] >= gaps[:-1] * 0.9)) if len(gaps) > 1 else 0.0
    if growing < 0.6:  # treads look bigger/further apart nearer the camera
        return None
    x1 = min(m[1] for m in merged)
    x2 = max(m[2] for m in merged)
    return {"box": [x1 / w, merged[0][0] / h, x2 / w, merged[-1][0] / h],
            "conf": round(min(0.6, 0.3 + 0.05 * len(merged)), 2)}


def _wall(gray, edges, w, h, vp_conf):
    x1, x2, y1, y2 = int(.25 * w), int(.75 * w), int(.30 * h), int(.80 * h)
    roi, e = gray[y1:y2, x1:x2], edges[y1:y2, x1:x2]
    density = float(np.count_nonzero(e)) / max(e.size, 1)
    std = float(roi.std())
    if density < 0.012 and std < 28 and vp_conf < 0.4:
        return {"box": [x1 / w, y1 / h, x2 / w, y2 / h],
                "conf": round(min(0.6, 0.3 + (0.012 - density) * 30), 2)}
    return None


def analyze_scene(bgr, width=320):
    h0, w0 = bgr.shape[:2]
    small = cv2.resize(bgr, (width, max(1, int(h0 * width / w0))))
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 140)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, minLineLength=int(w * 0.14), maxLineGap=8)
    lines = [] if raw is None else raw[:, 0, :].tolist()
    vp, vp_conf = _vanishing_point(lines, w, h)
    return {"vp": vp, "vp_conf": vp_conf, "stairs": _stairs(lines, w, h),
            "wall": _wall(gray, edges, w, h, vp_conf)}
