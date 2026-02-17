#!/usr/bin/env python3
"""
Stream Client – receives UDP video stream from stream_server.py, reassembles
frames, and serves a real-time web dashboard at http://localhost:5000

Features:
  • MJPEG video stream endpoint (/video_feed)
  • Live metrics via Server-Sent Events (/metrics_feed): FPS, latency, object count
  • Responsive dark-themed HTML dashboard
  • Graceful packet-loss handling

Usage:
    python stream_client.py                       # defaults
    python stream_client.py --port 9999 --web-port 5000
"""

import argparse
import json
import struct
import socket
import sys
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

import config

# ── Globals shared between UDP receiver thread and Flask ────────────────────

latest_frame: bytes | None = None        # JPEG bytes of most recent complete frame
frame_lock = threading.Lock()

# Metrics
metrics = {
    "fps": 0.0,
    "latency_ms": 0.0,
    "object_count": 0,
    "frames_received": 0,
    "frames_dropped": 0,
}
metrics_lock = threading.Lock()


# ── HTML Dashboard ──────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Detection &amp; Tracking – Live Stream</title>
<style>
  :root { --bg: #0f1117; --card: #1a1d28; --accent: #00e5ff; --text: #e0e0e0; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); }
  .header { text-align: center; padding: 18px 0 10px; }
  .header h1 { font-size: 1.5rem; font-weight: 600; color: var(--accent); letter-spacing: 1px; }
  .header p { font-size: 0.85rem; opacity: 0.6; margin-top: 2px; }
  .container { max-width: 1100px; margin: 0 auto; padding: 0 16px 30px; }
  .metrics { display: flex; gap: 14px; justify-content: center; margin-bottom: 16px; flex-wrap: wrap; }
  .metric-card {
    background: var(--card); border-radius: 10px; padding: 14px 24px; min-width: 140px;
    text-align: center; border: 1px solid rgba(255,255,255,0.06);
  }
  .metric-card .value { font-size: 1.8rem; font-weight: 700; color: var(--accent); }
  .metric-card .label { font-size: 0.75rem; text-transform: uppercase; opacity: 0.55; margin-top: 2px; letter-spacing: 1px; }
  .video-wrapper {
    border-radius: 12px; overflow: hidden; border: 2px solid rgba(0,229,255,0.15);
    box-shadow: 0 0 40px rgba(0,229,255,0.08); background: #000; text-align: center;
  }
  .video-wrapper img { width: 100%; max-width: 900px; display: block; margin: 0 auto; }
  .status { text-align: center; margin-top: 12px; font-size: 0.8rem; opacity: 0.5; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
  .dot.live { background: #00e676; animation: pulse 1.5s infinite; }
  .dot.offline { background: #ff1744; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
</style>
</head>
<body>
  <div class="header">
    <h1>REAL-TIME OBJECT DETECTION &amp; TRACKING</h1>
    <p>YOLOv8 + DeepSORT &middot; UDP Stream Receiver</p>
  </div>
  <div class="container">
    <div class="metrics">
      <div class="metric-card"><div class="value" id="fps">--</div><div class="label">FPS</div></div>
      <div class="metric-card"><div class="value" id="latency">--</div><div class="label">Latency (ms)</div></div>
      <div class="metric-card"><div class="value" id="objects">--</div><div class="label">Objects</div></div>
      <div class="metric-card"><div class="value" id="received">0</div><div class="label">Frames Recv</div></div>
      <div class="metric-card"><div class="value" id="dropped">0</div><div class="label">Dropped</div></div>
    </div>
    <div class="video-wrapper">
      <img id="stream" src="/video_feed" alt="Live stream" />
    </div>
    <div class="status" id="status"><span class="dot live"></span>Waiting for stream&hellip;</div>
  </div>

  <script>
    const es = new EventSource("/metrics_feed");
    es.onmessage = function(e) {
      const d = JSON.parse(e.data);
      document.getElementById("fps").textContent = d.fps.toFixed(1);
      document.getElementById("latency").textContent = d.latency_ms.toFixed(1);
      document.getElementById("objects").textContent = d.object_count;
      document.getElementById("received").textContent = d.frames_received;
      document.getElementById("dropped").textContent = d.frames_dropped;
      document.getElementById("status").innerHTML =
        '<span class="dot live"></span>Receiving from UDP ' + d.udp_port;
    };
    es.onerror = function() {
      document.getElementById("status").innerHTML =
        '<span class="dot offline"></span>Connection lost';
    };
  </script>
</body>
</html>
"""


# ── Flask App ───────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


def generate_mjpeg():
    """Yield MJPEG multipart frames for the <img> tag."""
    while True:
        with frame_lock:
            jpeg = latest_frame
        if jpeg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        else:
            # Send a tiny blank JPEG so the browser doesn't timeout
            blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for stream...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
            _, buf = cv2.imencode(".jpg", blank)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        time.sleep(0.03)  # ~33 FPS max


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/metrics_feed")
def metrics_feed():
    """Server-Sent Events endpoint for live metrics."""
    def event_stream():
        while True:
            with metrics_lock:
                data = dict(metrics)
            data["udp_port"] = udp_port_global
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.5)

    return Response(event_stream(), mimetype="text/event-stream")


# ── UDP Receiver Thread ─────────────────────────────────────────────────────

def udp_receiver(port: int):
    """Run in a daemon thread. Receives UDP packets, reassembles frames."""
    global latest_frame

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        sock.bind(("0.0.0.0", port))
        sock.settimeout(2.0)  # 2s timeout to allow clean shutdown
    except Exception as e:
        print(f"[client] ERROR binding UDP socket: {e}", flush=True)
        return

    print(f"[client] UDP receiver listening on 0.0.0.0:{port}", flush=True)

    frame_buffer: dict[int, dict] = {}   # frame_id -> {chunk_idx: bytes, ...}
    frame_meta: dict[int, dict] = {}     # frame_id -> {total_chunks, timestamp, obj_count}

    fps_times: deque[float] = deque(maxlen=60)
    last_cleanup = time.time()
    recv_count = 0

    while True:
      try:
        try:
            packet, addr = sock.recvfrom(config.MAX_DGRAM_SIZE)
        except socket.timeout:
            continue
        except OSError as e:
            print(f"[client] Socket OSError: {e}", flush=True)
            break

        recv_count += 1
        if recv_count <= 5 or recv_count % 100 == 0:
            print(f"[client] Received packet #{recv_count}, size={len(packet)} from {addr}", flush=True)

        if len(packet) < config.HEADER_SIZE:
            print(f"[client] Packet too small: {len(packet)} < {config.HEADER_SIZE}", flush=True)
            continue

        header = packet[: config.HEADER_SIZE]
        chunk_data = packet[config.HEADER_SIZE :]

        frame_id, obj_count, chunk_idx, total_chunks, timestamp = struct.unpack(
            config.HEADER_FORMAT, header
        )

        # Buffer chunk
        if frame_id not in frame_buffer:
            frame_buffer[frame_id] = {}
            frame_meta[frame_id] = {
                "total_chunks": total_chunks,
                "timestamp": timestamp,
                "obj_count": obj_count,
            }
        frame_buffer[frame_id][chunk_idx] = chunk_data

        # Check if frame is complete
        if len(frame_buffer[frame_id]) == frame_meta[frame_id]["total_chunks"]:
            total = frame_meta[frame_id]["total_chunks"]
            jpeg_data = b"".join(frame_buffer[frame_id][i] for i in range(total))

            # Decode to verify, then store JPEG bytes for MJPEG stream
            img = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                with frame_lock:
                    latest_frame = jpeg_data
                print(f"[client] Frame #{frame_id} decoded OK, {len(jpeg_data)} bytes, obj_count={obj_count}", flush=True)

                now = time.time()
                fps_times.append(now)
                latency = (now - timestamp) * 1000  # ms

                with metrics_lock:
                    metrics["object_count"] = obj_count
                    metrics["latency_ms"] = round(latency, 1)
                    metrics["frames_received"] += 1
                    if len(fps_times) >= 2:
                        elapsed = fps_times[-1] - fps_times[0]
                        metrics["fps"] = round((len(fps_times) - 1) / max(elapsed, 1e-6), 1)

            # Cleanup this frame
            del frame_buffer[frame_id]
            del frame_meta[frame_id]

        # Periodic stale-frame cleanup (every 1s)
        now_cleanup = time.time()
        if now_cleanup - last_cleanup > 1.0:
            stale_ids = [
                fid
                for fid, meta in frame_meta.items()
                if (now_cleanup - meta["timestamp"]) > 0.5
            ]
            for fid in stale_ids:
                frame_buffer.pop(fid, None)
                frame_meta.pop(fid, None)
                with metrics_lock:
                    metrics["frames_dropped"] += 1
            last_cleanup = now_cleanup
      except Exception as e:
        print(f"[client] UDP receiver error: {e}", flush=True)
        traceback.print_exc()
        break


# ── Entry point ─────────────────────────────────────────────────────────────

udp_port_global = config.UDP_PORT


def main():
    global udp_port_global

    parser = argparse.ArgumentParser(description="UDP stream client (web dashboard)")
    parser.add_argument("--port", type=int, default=config.UDP_PORT, help="UDP port to listen on")
    parser.add_argument("--web-port", type=int, default=config.WEB_PORT, help="Web dashboard port")
    parser.add_argument("--web-host", default=config.WEB_HOST, help="Web dashboard host")
    args = parser.parse_args()

    udp_port_global = args.port

    # Start UDP receiver in background thread
    recv_thread = threading.Thread(target=udp_receiver, args=(args.port,), daemon=True)
    recv_thread.start()

    print(f"[client] Web dashboard → http://localhost:{args.web_port}", flush=True)
    app.run(host=args.web_host, port=args.web_port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
