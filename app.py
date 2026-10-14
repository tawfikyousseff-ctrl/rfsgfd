"""AR indoor navigation (no GPS / no markers) - Flask server.
Run:  python app.py      then open https://<PC-IP>:8443 on the phone.
"""
import os
import time
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from detector import Detector
from guidance import build_guidance
from heuristics import analyze_scene
from make_cert import ensure_cert, lan_ips
from visual_odometry import VisualOdometry

PORT = int(os.environ.get("PORT", 8443))

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
detector = None
sessions = {}
sessions_lock = threading.Lock()


def get_session(sid):
    now = time.time()
    with sessions_lock:
        for k in [k for k, v in sessions.items() if now - v["t"] > 600]:
            del sessions[k]
        s = sessions.setdefault(sid, {"vo": VisualOdometry(), "t": now})
        s["t"] = now
        return s


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify(ok=True, weights=detector.weights if detector else "none")


@app.route("/api/reset", methods=["POST"])
def reset():
    get_session(request.headers.get("X-Session", "anon"))["vo"].reset()
    return jsonify(ok=True)


@app.route("/api/process", methods=["POST"])
def process():
    t0 = time.time()
    sid = request.headers.get("X-Session", "anon")
    frame = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="bad image"), 400
    h, w = frame.shape[:2]

    dets = detector.detect(frame)
    scene = analyze_scene(frame)
    # heuristic stairs / wall only when the model did not already find them
    for kind, label in (("stairs", "سلم (تقدير)"), ("wall", "جدار (تقدير)")):
        cue = scene.get(kind)
        if cue and not any(d["kind"] == kind for d in dets):
            dets.append({"kind": kind, "label": label, "conf": cue["conf"], "src": "cv", "box": cue["box"]})

    vo = get_session(sid)["vo"].update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    g = build_guidance(dets, scene, w / h)
    return jsonify(detections=dets, vp=scene["vp"], vp_conf=scene["vp_conf"], vo=vo, guidance=g,
                   size=[w, h], ms=int((time.time() - t0) * 1000))


if __name__ == "__main__":
    cert, key = ensure_cert()
    detector = Detector()
    print("\nOpen on your phone (same Wi-Fi):")
    for ip in lan_ips():
        print(f"   https://{ip}:{PORT}")
    print("(accept the security warning once - the certificate is self-signed)\n")
    app.run(host="0.0.0.0", port=PORT, ssl_context=(cert, key), threaded=True, debug=False)
