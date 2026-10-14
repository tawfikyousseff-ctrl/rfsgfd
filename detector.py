"""YOLO wrapper. Maps raw model classes to: door / stairs / wall / obstacle.

Default weights: yolov8n-oiv7.pt (Open Images V7, 600 classes, includes Door and
Stairs). If it cannot be loaded, falls back to COCO yolov8n.pt (obstacles only).
You can also drop your own trained model in models/indoor.pt (classes such as
door, stairs, wall are recognised by name).
"""
import os
import threading

OBSTACLES = {
    "person": "شخص", "man": "شخص", "woman": "شخص", "boy": "طفل", "girl": "طفلة", "child": "طفل",
    "chair": "كرسي", "stool": "كرسي", "bench": "مقعد", "couch": "كنبة", "sofa bed": "كنبة",
    "bed": "سرير", "table": "طاولة", "desk": "مكتب", "coffee table": "طاولة",
    "kitchen & dining room table": "طاولة", "dining table": "طاولة", "box": "صندوق",
    "cabinetry": "دولاب", "filing cabinet": "دولاب", "shelf": "رف", "bookcase": "مكتبة",
    "trash can": "سلة مهملات", "plant": "نبات", "houseplant": "نبات", "potted plant": "نبات",
    "suitcase": "شنطة", "luggage and bags": "شنطة", "backpack": "شنطة", "bicycle": "عجلة",
    "wheelchair": "كرسي متحرك", "television": "تلفزيون", "tv": "تلفزيون",
    "refrigerator": "ثلاجة", "dog": "كلب", "cat": "قطة", "ladder": "سلم متنقل",
    "cart": "عربة", "stroller": "عربة أطفال", "furniture": "أثاث",
}
DOORS = {"door": "باب"}
STAIRS = {"stairs": "سلم", "staircase": "سلم"}
WALLS = {"wall": "جدار"}

MIN_CONF = {"door": 0.25, "stairs": 0.25, "wall": 0.30, "obstacle": 0.35}


class Detector:
    def __init__(self, imgsz=480):
        self.lock = threading.Lock()
        self.imgsz = imgsz
        self.model = None
        self.weights = "none"
        if os.environ.get("DETECTOR", "yolo").lower() == "none":
            print("[detector] disabled (DETECTOR=none)")
            return
        from ultralytics import YOLO  # imported lazily so the rest can be tested without torch

        here = os.path.dirname(os.path.abspath(__file__))
        custom = os.path.join(here, "models", "indoor.pt")
        candidates = []
        if os.path.exists(custom):
            candidates.append(custom)
        candidates += [os.environ.get("YOLO_WEIGHTS", "yolov8n-oiv7.pt"), "yolov8n.pt"]
        for w in candidates:
            try:
                self.model = YOLO(w)
                self.weights = os.path.basename(w)
                print(f"[detector] loaded {w}")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[detector] could not load {w}: {exc}")
        if self.model is None:
            raise RuntimeError("No YOLO weights could be loaded (need internet for the first download).")

    def detect(self, bgr):
        if self.model is None:
            return []
        h, w = bgr.shape[:2]
        with self.lock:
            res = self.model.predict(bgr, imgsz=self.imgsz, conf=0.25, verbose=False)[0]
        out = []
        for b in res.boxes:
            name = str(res.names[int(b.cls)]).lower()
            conf = float(b.conf)
            if name in DOORS:
                kind, label = "door", DOORS[name]
            elif name in STAIRS:
                kind, label = "stairs", STAIRS[name]
            elif name in WALLS:
                kind, label = "wall", WALLS[name]
            elif name in OBSTACLES:
                kind, label = "obstacle", OBSTACLES[name]
            else:
                continue
            if conf < MIN_CONF[kind]:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append({"kind": kind, "label": label, "conf": round(conf, 2), "src": "yolo",
                        "box": [x1 / w, y1 / h, x2 / w, y2 / h]})
        return out
