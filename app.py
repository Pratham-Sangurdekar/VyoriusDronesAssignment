#!/usr/bin/env python3
"""
Real-Time Object Detection & Tracking – Web Live View

Single application that:
  1. Captures webcam (or video file)
  2. Runs YOLOv8 object detection
  3. Tracks objects with DeepSORT (persistent IDs + trajectories)
  4. Serves annotated live video at http://localhost:5050

Usage:
    python app.py                        # webcam
    python app.py --source video.mp4     # video file
    python app.py --port 8080            # custom port
"""

import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

import config
from detector import ObjectDetector
from tracker import ObjectTracker

# ─── Globals ────────────────────────────────────────────────────────────────

latest_jpeg: bytes | None = None
jpeg_lock = threading.Lock()

live_metrics = {"fps": 0.0, "objects": 0, "frame_count": 0}
metrics_lock = threading.Lock()

# ─── Visualisation helpers ──────────────────────────────────────────────────

def get_color(class_name: str) -> tuple[int, int, int]:
    return config.CLASS_COLORS.get(class_name, config.DEFAULT_COLOR)


def draw_tracks(frame, tracks: list[dict], trajectories: dict):
    for t in tracks:
        tid = t["track_id"]
        l, top, r, b = t["bbox_ltrb"]
        cls = t["class_name"] or "?"
        conf = t["confidence"]
        color = get_color(cls)

        # Bounding box
        cv2.rectangle(frame, (l, top), (r, b), color, 2)

        # Label
        conf_str = f"{conf:.2f}" if conf is not None else "---"
        label = f"ID:{tid} {cls} {conf_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (l, top - th - 8), (l + tw + 4, top), color, -1)
        cv2.putText(frame, label, (l + 2, top - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Trajectory
        pts = trajectories.get(tid)
        if pts and len(pts) > 1:
            arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [arr], False, color, 2, cv2.LINE_AA)


def draw_hud(frame, fps: float, obj_count: int):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {obj_count}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


# ─── Detection + Tracking pipeline (runs in background thread) ─────────────

def pipeline(source):
    global latest_jpeg

    print(f"[pipeline] Loading detector ({config.YOLO_MODEL}) ...", flush=True)
    detector = ObjectDetector()

    print(f"[pipeline] Initialising tracker (embedder={config.TRACKER_EMBEDDER}) ...", flush=True)
    tracker = ObjectTracker()

    print(f"[pipeline] Opening video source: {source}", flush=True)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[pipeline] ERROR: Cannot open '{source}'", flush=True)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    fps_deque: deque[float] = deque(maxlen=30)
    prev = time.perf_counter()
    frame_count = 0

    print("[pipeline] Running — view at http://localhost:5050", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        # Detect
        detections = detector.detect(frame)

        # Track
        tracks = tracker.update(detections, frame)
        trajectories = tracker.get_trajectories()

        # Annotate
        draw_tracks(frame, tracks, trajectories)
        now = time.perf_counter()
        fps_deque.append(1.0 / max(now - prev, 1e-6))
        prev = now
        fps = sum(fps_deque) / len(fps_deque)
        draw_hud(frame, fps, len(tracks))

        # Encode to JPEG and publish
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with jpeg_lock:
            latest_jpeg = buf.tobytes()

        frame_count += 1
        with metrics_lock:
            live_metrics["fps"] = round(fps, 1)
            live_metrics["objects"] = len(tracks)
            live_metrics["frame_count"] = frame_count

        if frame_count <= 3 or frame_count % 100 == 0:
            print(f"[pipeline] frame={frame_count}  fps={fps:.1f}  objects={len(tracks)}", flush=True)

    cap.release()
    print("[pipeline] Stopped.", flush=True)


# ─── Flask web app ──────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Detection &amp; Tracking – Live</title>
<style>
  :root { --bg:#0f1117; --card:#1a1d28; --accent:#00e5ff; --text:#e0e0e0; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text); }
  .header { text-align:center; padding:18px 0 10px; }
  .header h1 { font-size:1.5rem; font-weight:600; color:var(--accent); letter-spacing:1px; }
  .header p { font-size:0.85rem; opacity:0.6; margin-top:2px; }
  .container { max-width:1100px; margin:0 auto; padding:0 16px 30px; }
  .metrics { display:flex; gap:14px; justify-content:center; margin-bottom:16px; flex-wrap:wrap; }
  .metric-card {
    background:var(--card); border-radius:10px; padding:14px 24px; min-width:140px;
    text-align:center; border:1px solid rgba(255,255,255,0.06);
  }
  .metric-card .value { font-size:1.8rem; font-weight:700; color:var(--accent); }
  .metric-card .label { font-size:0.75rem; text-transform:uppercase; opacity:0.55; margin-top:2px; letter-spacing:1px; }
  .video-wrapper {
    border-radius:12px; overflow:hidden; border:2px solid rgba(0,229,255,0.15);
    box-shadow:0 0 40px rgba(0,229,255,0.08); background:#000; text-align:center;
  }
  .video-wrapper img { width:100%; max-width:900px; display:block; margin:0 auto; }
  .status { text-align:center; margin-top:12px; font-size:0.8rem; opacity:0.5; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }
  .dot.live { background:#00e676; animation:pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
</style>
</head>
<body>
  <div class="header">
    <h1>REAL-TIME OBJECT DETECTION &amp; TRACKING</h1>
    <p>YOLOv8 + DeepSORT &middot; Live Feed</p>
  </div>
  <div class="container">
    <div class="metrics">
      <div class="metric-card"><div class="value" id="fps">--</div><div class="label">FPS</div></div>
      <div class="metric-card"><div class="value" id="objects">--</div><div class="label">Objects</div></div>
      <div class="metric-card"><div class="value" id="frames">0</div><div class="label">Frames</div></div>
    </div>
    <div class="video-wrapper">
      <img id="stream" src="/video_feed" alt="Live stream"/>
    </div>
    <div class="status"><span class="dot live"></span>Live</div>
  </div>
  <script>
    setInterval(function(){
      fetch("/metrics").then(r=>r.json()).then(d=>{
        document.getElementById("fps").textContent=d.fps.toFixed(1);
        document.getElementById("objects").textContent=d.objects;
        document.getElementById("frames").textContent=d.frame_count;
      }).catch(()=>{});
    }, 1000);
  </script>
</body>
</html>
"""

app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


def generate_mjpeg():
    while True:
        with jpeg_lock:
            jpeg = latest_jpeg
        if jpeg is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        else:
            blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank, "Loading models...", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
            _, buf = cv2.imencode(".jpg", blank)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/metrics")
def metrics_endpoint():
    import json
    with metrics_lock:
        return json.dumps(live_metrics), 200, {"Content-Type": "application/json"}


# ─── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live detection & tracking viewer")
    parser.add_argument("--source", default=None, help="0 for webcam, or path to video file")
    parser.add_argument("--port", type=int, default=config.WEB_PORT)
    parser.add_argument("--embedder", default=config.TRACKER_EMBEDDER,
                        help="'mobilenet' or 'none'")
    args = parser.parse_args()

    source = config.INPUT_SOURCE if args.source is None else args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    if str(args.embedder).lower() == "none":
        config.TRACKER_EMBEDDER = None
    else:
        config.TRACKER_EMBEDDER = args.embedder

    # Start the detection pipeline in a background thread
    t = threading.Thread(target=pipeline, args=(source,), daemon=True)
    t.start()

    print(f"\n  Open http://localhost:{args.port} in your browser\n", flush=True)
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
