"""Turns detections + scene cues into one Arabic instruction and a steering angle."""
import math

ZONES = {"left": (0.0, 0.36), "center": (0.36, 0.64), "right": (0.64, 1.0)}


def _ov(x1, x2, a, b):
    return max(0.0, min(x2, b) - max(x1, a)) / (b - a)


def build_guidance(dets, scene, aspect):
    f = 0.95 if aspect < 1 else 0.80  # focal length in units of image width (matches VO guess)
    ang = lambda nx: math.degrees(math.atan((nx - 0.5) / f))  # noqa: E731

    score = {"left": 0.0, "center": 0.0, "right": 0.0}
    near = False
    for d in dets:
        if d["kind"] != "obstacle":
            continue
        x1, y1, x2, y2 = d["box"]
        if y2 < 0.5:
            continue
        closeness = max(y2 - 0.5, 0) * 2 + (y2 - y1)
        for z, (a, b) in ZONES.items():
            o = _ov(x1, x2, a, b)
            score[z] += o * closeness
            if z == "center" and o > 0.3 and (y2 > 0.82 or (y2 - y1) > 0.5):
                near = True

    def in_center(kind, min_bottom=0.0):
        hits = [d for d in dets if d["kind"] == kind and _ov(d["box"][0], d["box"][2], *ZONES["center"]) > 0.3
                and d["box"][3] > min_bottom]
        return max(hits, key=lambda d: d["box"][3]) if hits else None

    free = "right" if score["right"] <= score["left"] else "left"
    side_ar = "يمين" if free == "right" else "يسار"

    stairs, wall, door = in_center("stairs", 0.45), in_center("wall", 0.5), in_center("door")

    if score["center"] > 0.30:
        out = {"code": "obstacle", "text": "خلي بالك من العائق", "hint": f"لف {side_ar}",
               "level": "danger" if near else "warn"}
        steer = 35 if free == "right" else -35
    elif stairs:
        out = {"code": "stairs", "text": "السلم أمامك", "hint": "تمهّل وخلي بالك",
               "level": "danger" if stairs["box"][3] > 0.8 else "warn"}
        steer = 0
    elif wall:
        out = {"code": "wall", "text": "جدار أمامك", "hint": "لف يمين أو يسار", "level": "warn"}
        steer = 0
    elif door:
        cx = (door["box"][0] + door["box"][2]) / 2
        out = {"code": "door", "text": "الباب أمامك", "hint": "", "level": "info"}
        steer = ang(cx)
    else:
        vp, conf = scene.get("vp"), scene.get("vp_conf", 0)
        steer = ang(vp[0]) if (vp and conf >= 0.4) else 0.0
        if steer > 12:
            out = {"code": "right", "text": "لف يمين", "hint": "الممر يميل لليمين", "level": "info"}
        elif steer < -12:
            out = {"code": "left", "text": "لف يسار", "hint": "الممر يميل لليسار", "level": "info"}
        else:
            out = {"code": "forward", "text": "امشِ للأمام",
                   "hint": "الممر أمامك" if conf >= 0.4 else "", "level": "ok"}
    return {"default": out, "steer": round(max(-60.0, min(60.0, steer)), 1),
            "zones": {k: round(v, 2) for k, v in score.items()}}
