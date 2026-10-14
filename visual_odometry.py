"""Monocular visual odometry between consecutive frames (ORB + essential matrix).

What it gives: relative camera ROTATION (yaw), a tracking-quality score (inliers) and
a 'is the camera translating' flag. Translation SCALE is unobservable with one camera,
so distance is NOT taken from here (the browser uses step counting for scale).
"""
import math

import cv2
import numpy as np


class VisualOdometry:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_des = None

    def reset(self):
        self.prev_kp = self.prev_des = None

    def update(self, gray):
        h, w = gray.shape[:2]
        f = (0.95 if h > w else 0.80) * w  # rough focal length guess (px); phones are ~65-75 deg HFOV
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
        kp, des = self.orb.detectAndCompute(gray, None)
        res = {"ok": False, "yaw": 0.0, "inliers": 0, "flow": 0.0, "moving": False, "features": len(kp) if kp else 0}
        prev_kp, prev_des = self.prev_kp, self.prev_des
        self.prev_kp, self.prev_des = kp, des
        if prev_des is None or des is None or len(kp) < 30 or len(prev_kp) < 30:
            return res
        matches = sorted(self.bf.match(prev_des, des), key=lambda m: m.distance)[:400]
        if len(matches) < 40:
            return res
        p1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        p2 = np.float32([kp[m.trainIdx].pt for m in matches])
        flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
        res["flow"] = round(flow, 2)
        res["moving"] = flow > 0.8
        try:
            E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
            if E is None:
                return res
            E = E[:3]
            n, R, t, mask2 = cv2.recoverPose(E, p1, p2, K, mask=mask)
        except cv2.error:
            return res
        rvec, _ = cv2.Rodrigues(R)
        # camera axes: x right, y down, z forward. Turning right = negative rotation about y of the points.
        yaw = -math.degrees(float(rvec[1][0]))
        res["inliers"] = int(n)
        res["ok"] = n >= 25
        res["yaw"] = round(yaw if flow > 0.5 else 0.0, 2)
        return res
