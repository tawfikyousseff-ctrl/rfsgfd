# AR Indoor Navigation (no GPS, no markers)

Flask + OpenCV + YOLO on your Windows PC; your phone's rear camera streams frames over HTTPS.

## Install (Windows, Python 3.10-3.12)
    cd ar-indoor-nav
    python -m venv venv
    venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python app.py

(or double-click run.bat). First start downloads YOLO weights (needs internet once).

## Open on the phone
1. PC and phone on the same Wi-Fi. Set the Wi-Fi network to *Private* in Windows.
2. Allow the port (PowerShell as admin):
   `netsh advfirewall firewall add rule name="AR Nav" dir=in action=allow protocol=TCP localport=8443`
3. The console prints `https://<PC-IP>:8443`. Open it in Chrome (Android) or Safari (iPhone).
4. Accept the self-signed certificate warning (Chrome: Advanced -> Proceed).
5. Tap "ابدأ الكاميرا", allow camera + motion sensors. Hold the phone upright, camera forward.
If the IP changes: `python make_cert.py --force`, then restart.

## How it works
- YOLOv8n Open Images V7 (`yolov8n-oiv7.pt`): Door, Stairs + common obstacles (people, chairs, tables...).
  Put your own model in `models/indoor.pt` (class names door / stairs / wall) to improve accuracy.
- OpenCV heuristics: corridor vanishing point, stair-like repeated edges (dashed boxes), blank-wall cue.
- Positioning: step counting + gyro heading (pedestrian dead reckoning) on the phone, with a gentle
  correction from monocular visual odometry (ORB + essential matrix) on the server.
- Save places while walking; later tap "اذهب" for arrow/voice guidance toward them.

## Limitations (read this)
- NOT a reliable navigation system. No map of the building, no loop closure: position drifts (typically
  several % of distance walked plus heading error). Saved places stay valid only if you start from the
  same spot, facing the same direction.
- Monocular VO has no metric scale; distance comes from step counting (set your step length).
- Doors/stairs/walls detection is approximate (pretrained model + heuristics): expect misses and false alarms.
  Never rely on it for stairs, traffic, or accessibility safety.
- Boxes lag the live video by the network+inference delay (~0.1-0.4 s). Arabic voice needs an Arabic TTS voice on the phone.
