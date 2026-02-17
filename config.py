"""
Configuration for the real-time object detection, tracking & UDP streaming system.
All tunables are centralised here. Override via CLI args where supported.
"""

# ── Video Input ──────────────────────────────────────────────────────────────
# 0 for webcam, or a path string like "sample.mp4"
INPUT_SOURCE = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ── YOLOv8 Detection ────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"          # auto-downloads on first run (~6 MB)
YOLO_CONF = 0.4                    # confidence threshold
YOLO_IOU = 0.7                     # NMS IoU threshold
YOLO_IMGSZ = 320                   # inference resolution (320 = fast, 640 = accurate)
YOLO_DEVICE = ""                   # "" = auto (MPS on Apple Silicon, CUDA if available, else CPU)

# COCO class IDs to detect:
#   Vehicles: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
#   Indoor:   39=bottle, 41=cup, 56=chair, 57=couch, 58=potted plant,
#             60=dining table, 62=tv, 63=laptop, 64=mouse, 65=remote,
#             66=keyboard, 67=cell phone, 73=book, 74=clock
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 39, 41, 56, 57, 58, 60, 62, 63, 64, 65, 66, 67, 73, 74]

# ── DeepSORT Tracking ───────────────────────────────────────────────────────
TRACKER_MAX_AGE = 30               # frames a lost track survives (occlusion tolerance)
TRACKER_N_INIT = 3                 # detections before a track is confirmed
TRACKER_MAX_IOU_DISTANCE = 0.7
TRACKER_MAX_COSINE_DISTANCE = 0.2
TRACKER_NN_BUDGET = 100

# Set to "mobilenet" for appearance-based tracking (robust, ~10ms overhead)
# Set to None for pure IoU tracking (fast, less robust to occlusion)
TRACKER_EMBEDDER = "mobilenet"

# Number of past centre-points to draw as trajectory tail
TRAJECTORY_LENGTH = 30

# ── UDP Streaming Protocol ──────────────────────────────────────────────────
UDP_IP = "127.0.0.1"
UDP_PORT = 9999

JPEG_QUALITY = 70                  # 0-100; 70 ≈ 25-40 KB at 640×480

# Packet structure
import struct
MAX_DGRAM_SIZE = 9216              # macOS default; use 65507 on Linux
# Header: frame_id(I) | object_count(H) | chunk_index(H) | total_chunks(H) | timestamp(d)
#   4 + 2 + 2 + 2 + 8 = 18 bytes
HEADER_FORMAT = "!IHHHd"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)   # 18 bytes
MAX_CHUNK_SIZE = MAX_DGRAM_SIZE - HEADER_SIZE

# ── Display / Server ────────────────────────────────────────────────────────
DISPLAY_LOCAL = True               # show annotated video on the server side (cv2.imshow)

# ── Web Client ──────────────────────────────────────────────────────────────
WEB_HOST = "0.0.0.0"
WEB_PORT = 5050                    # http://localhost:5050

# ── Visualisation Colors (BGR) ──────────────────────────────────────────────
CLASS_COLORS = {
    "person":     (0, 255, 0),
    "bicycle":    (255, 165, 0),
    "car":        (0, 120, 255),
    "motorcycle": (255, 0, 255),
    "bus":        (0, 255, 255),
    "truck":      (255, 255, 0),
    "bottle":     (128, 0, 255),
    "cup":        (255, 128, 0),
    "chair":      (0, 200, 200),
    "couch":      (200, 100, 50),
    "potted plant": (0, 180, 0),
    "dining table": (180, 180, 0),
    "tv":         (255, 0, 128),
    "laptop":     (100, 200, 255),
    "mouse":      (200, 200, 200),
    "remote":     (150, 100, 255),
    "keyboard":   (100, 255, 200),
    "cell phone": (255, 100, 100),
    "book":       (180, 130, 70),
    "clock":      (50, 200, 150),
}
DEFAULT_COLOR = (200, 200, 200)
