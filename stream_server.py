#!/usr/bin/env python3
"""
Stream Server – captures video, runs detection + tracking, annotates frames,
and streams the result over UDP to the client.

Usage:
    python stream_server.py                       # webcam
    python stream_server.py --source video.mp4    # video file
    python stream_server.py --no-display          # headless (no local window)
"""

import argparse
import math
import socket
import struct
import sys
import time
from collections import deque

import cv2
import numpy as np

import config
from detector import ObjectDetector
from tracker import ObjectTracker


# ── Annotation helpers ──────────────────────────────────────────────────────

def get_color(class_name: str) -> tuple[int, int, int]:
    return config.CLASS_COLORS.get(class_name, config.DEFAULT_COLOR)


def draw_tracks(frame, tracks: list[dict], trajectories: dict):
    """Draw bounding boxes, labels, and trajectory tails on the frame."""
    for t in tracks:
        tid = t["track_id"]
        l, top, r, b = t["bbox_ltrb"]
        cls = t["class_name"] or "?"
        conf = t["confidence"]
        color = get_color(cls)

        # Bounding box
        cv2.rectangle(frame, (l, top), (r, b), color, 2)

        # Label background + text
        conf_str = f"{conf:.2f}" if conf is not None else "---"
        label = f"ID:{tid} {cls} {conf_str}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (l, top - th - 8), (l + tw + 4, top), color, -1)
        cv2.putText(frame, label, (l + 2, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Trajectory polyline
        pts = trajectories.get(tid)
        if pts and len(pts) > 1:
            arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [arr], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)


def draw_hud(frame, fps: float, obj_count: int):
    """Draw FPS and object count at the top-left."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {obj_count}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


# ── UDP sender ──────────────────────────────────────────────────────────────

def send_frame_udp(sock, frame, frame_id: int, obj_count: int, addr: tuple):
    """JPEG-encode a frame, fragment it, and send over UDP."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    data = buf.tobytes()
    total_chunks = max(1, math.ceil(len(data) / config.MAX_CHUNK_SIZE))
    timestamp = time.time()

    for i in range(total_chunks):
        start = i * config.MAX_CHUNK_SIZE
        end = start + config.MAX_CHUNK_SIZE
        chunk = data[start:end]
        header = struct.pack(
            config.HEADER_FORMAT,
            frame_id,
            obj_count,
            i,               # chunk_index
            total_chunks,
            timestamp,
        )
        try:
            sock.sendto(header + chunk, addr)
        except OSError:
            pass  # client not listening yet – fire and forget


# ── Main loop ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detection + Tracking stream server")
    parser.add_argument("--source", default=None, help="Video source: 0 for webcam, or path to file")
    parser.add_argument("--port", type=int, default=config.UDP_PORT, help="UDP port")
    parser.add_argument("--ip", default=config.UDP_IP, help="UDP destination IP")
    parser.add_argument("--no-display", action="store_true", help="Disable local preview window")
    parser.add_argument("--embedder", default=config.TRACKER_EMBEDDER,
                        help="DeepSORT embedder: 'mobilenet' or 'none'")
    args = parser.parse_args()

    # Resolve source
    source = config.INPUT_SOURCE if args.source is None else args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # it's a file path string

    display = config.DISPLAY_LOCAL and not args.no_display
    embedder = None if str(args.embedder).lower() == "none" else args.embedder
    udp_addr = (args.ip, args.port)

    # ── Initialise components ────────────────────────────────────────────
    print(f"[server] Loading detector ({config.YOLO_MODEL}) ...")
    detector = ObjectDetector()

    print(f"[server] Initialising tracker (embedder={embedder}) ...")
    tracker = ObjectTracker(embedder=embedder)

    print(f"[server] Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[server] ERROR: Cannot open video source '{source}'", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    frame_id = 0
    fps_deque: deque[float] = deque(maxlen=30)
    prev_time = time.perf_counter()

    print(f"[server] Streaming to UDP {udp_addr[0]}:{udp_addr[1]}")
    if display:
        print("[server] Local preview window enabled (press 'q' to quit)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video files; break for webcam EOF
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

            # FPS
            now = time.perf_counter()
            fps_deque.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            fps = sum(fps_deque) / len(fps_deque)

            draw_hud(frame, fps, len(tracks))

            # Stream
            send_frame_udp(sock, frame, frame_id, len(tracks), udp_addr)
            frame_id += 1

            # Local display
            if display:
                cv2.imshow("Server – Detection & Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[server] Interrupted.")
    finally:
        cap.release()
        sock.close()
        if display:
            cv2.destroyAllWindows()
        print("[server] Shut down.")


if __name__ == "__main__":
    main()
