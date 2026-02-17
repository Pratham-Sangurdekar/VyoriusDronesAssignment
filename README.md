<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-blue?logo=yolo&logoColor=white" />
  <img src="https://img.shields.io/badge/DeepSORT-Tracking-green" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-yellow?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-Web%20UI-lightgrey?logo=flask" />
  <img src="https://img.shields.io/badge/OpenCV-Video-red?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/UDP-Streaming-orange" />
</p>

<h1 align="center">Real-Time Object Detection & Tracking System</h1>

<p align="center">
  <img src="demo.gif" alt="Demo – real-time object detection & tracking" width="720" />
</p>

<p align="center">
  <b>YOLOv8 detection · DeepSORT tracking · UDP video streaming · Live web dashboard</b>
</p>

<p align="center">
  A complete computer-vision pipeline that captures video from a webcam or file,<br/>
  detects objects in real time, tracks them with persistent IDs & trajectory trails,<br/>
  and streams the annotated feed to a sleek browser-based dashboard.
</p>



## Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Run](#quick-run)
- [Usage Guide](#-usage-guide)
  - [Live Web View (Recommended)](#-live-web-view-recommended)
  - [UDP Streaming Mode](#-udp-streaming-mode)
  - [CLI Options](#cli-options)
- [Configuration Reference](#-configuration-reference)
- [How It Works](#-how-it-works)
  - [Object Detection](#1-object-detection)
  - [Multi-Object Tracking](#2-multi-object-tracking)
  - [Annotation & Visualisation](#3-annotation--visualisation)
  - [UDP Streaming Protocol](#4-udp-streaming-protocol)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)



## Features

| Feature | Description |
|:--------|:-----------|
| **Real-Time Detection** | YOLOv8n (nano) running at **15-30 FPS** on CPU — detects 6 object classes out of the box |
| **Persistent Tracking** | DeepSORT with MobileNet appearance embedder assigns **unique IDs** that survive occlusion |
| **Trajectory Trails** | Last 30 frame positions drawn as coloured polylines so you can see movement paths |
| **Live Web Dashboard** | Beautiful dark-themed browser UI at `localhost:5050` with MJPEG stream + live metrics |
| **UDP Video Streaming** | Server → Client architecture with custom fragmented packet protocol |
| **Fully Configurable** | Single `config.py` controls everything — model, thresholds, classes, ports, colours |
| **Occlusion Handling** | Kalman filter prediction keeps tracks alive for 30 frames when objects are temporarily hidden |
| **Live Metrics** | FPS, object count, and frame counter updated in real time on the web UI |

### Detected Object Classes

| Class | COCO ID | Box Colour |
|:------|:--------|:-----------|
| Person | 0 | Green |
| Bicycle | 1 | Orange |
| Car | 2 | Blue |
| Motorcycle | 3 | Magenta |
| Bus | 5 | Cyan |
| Truck | 7 | Yellow |

> Easily add more by editing `TARGET_CLASSES` in `config.py` — YOLO supports all 80 COCO classes!



## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         app.py (Web Mode)                       │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐      │
│  │ OpenCV   │    │ detector.py  │    │   tracker.py     │      │
│  │ Capture  │───►│ YOLOv8n      │───►│   DeepSORT       │      │
│  │ (webcam) │    │ Detection    │    │   Tracking + IDs │      │
│  └──────────┘    └──────────────┘    └────────┬─────────┘      │
│                                               │                 │
│                                        annotate frame           │
│                                        (boxes, labels,          │
│                                         trajectories, HUD)      │
│                                               │                 │
│                                        JPEG encode              │
│                                               │                 │
│                                    ┌──────────┴──────────┐      │
│                                    │   Flask MJPEG Server │      │
│                                    │   localhost:5050     │      │
│                                    └──────────┬──────────┘      │
└───────────────────────────────────────────────┤─────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────┐
                                    │      Browser       │
                                    │   Live Dashboard   │
                                    │   • Video stream   │
                                    │   • FPS counter    │
                                    │   • Object count   │
                                    └───────────────────┘
```

### UDP Streaming Mode (Alternative)

```
stream_server.py                          stream_client.py
┌──────────────┐    UDP packets           ┌──────────────┐
│ Detect+Track │ ──────────────────────►  │ Reassemble   │
│ Annotate     │   [hdr|JPEG chunk]       │ Decode       │
│ Compress     │   port 9999              │ Flask Web UI │
└──────────────┘                          └──────┬───────┘
                                                 │
                                          localhost:5050
```



## Tech Stack

| Component | Technology | Purpose |
|:----------|:-----------|:--------|
| **Detection** | [Ultralytics YOLOv8n](https://github.com/ultralytics/ultralytics) | Real-time object detection (3.2M params, 8.7 GFLOPs) |
| **Tracking** | [DeepSORT](https://github.com/levan92/deep_sort_realtime) | Multi-object tracking with appearance features |
| **Embedder** | MobileNetV2 (via torchvision) | Visual appearance encoding for track re-identification |
| **Video I/O** | [OpenCV](https://opencv.org/) | Webcam capture, frame processing, JPEG encoding |
| **Web UI** | [Flask](https://flask.palletsprojects.com/) | MJPEG streaming server + REST metrics endpoint |
| **Protocol** | Custom UDP | Fragmented packet streaming with metadata headers |
| **Language** | Python 3.10+ | Everything |



## Project Structure

```
VyoriusDronesAssignment/
│
├── app.py               # Main entry — live web view (recommended)
├── detector.py          # ObjectDetector class (YOLOv8 wrapper)
├── tracker.py           # ObjectTracker class (DeepSORT + trajectories)
├── config.py            #  All configuration in one place
│
├── stream_server.py     # UDP streaming server (alternative mode)
├── stream_client.py     # UDP streaming client + web dashboard
│
├── requirements.txt     # Python dependencies
└── README.md            # You are here!
```

### Module Breakdown

| Module | Class / Role | Key Methods |
|:-------|:-------------|:------------|
| `detector.py` | `ObjectDetector` | `detect(frame)` → list of `{bbox, confidence, class_name, class_id}` |
| `tracker.py` | `ObjectTracker` | `update(detections, frame)` → list of `{track_id, bbox_ltrb, class_name, confidence}` |
| | | `get_trajectories()` → `{track_id: [(cx,cy), ...]}` |
| `config.py` | Constants | All tunables: model, thresholds, classes, ports, colours |
| `app.py` | Pipeline + Flask | Background thread: capture→detect→track→annotate→JPEG; Flask serves MJPEG |



## Getting Started

### Prerequisites

- **Python 3.10+** (tested on 3.12 & 3.14)
- **Webcam** (built-in or USB) — or a video file
- ~500 MB disk space (for PyTorch + model weights on first run)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Pratham-Sangurdekar/VyoriusDronesAssignment.git
cd VyoriusDronesAssignment

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** On first run, YOLOv8n weights (~6 MB) and MobileNet embedder weights will be auto-downloaded.

### Quick Run

```bash
python app.py
```

Then open **http://localhost:5050** in your browser. That's it! 🎉



## 📖 Usage Guide

### Live Web View (Recommended)

The simplest way — one command, everything in your browser:

```bash
# Webcam (default)
python app.py

# Video file
python app.py --source path/to/video.mp4

# Custom port
python app.py --port 8080

# Faster mode (pure IoU tracking, no appearance features)
python app.py --embedder none
```

Open **http://localhost:5050** (or your custom port) and you'll see:
- Live annotated video with bounding boxes, track IDs, and trajectory trails
- Real-time FPS, object count, and frame counter
- Dark-themed responsive dashboard

### UDP Streaming Mode

For a distributed setup (server on one machine, client on another):

**Terminal 1 — Start the client (receiver + web UI):**
```bash
python stream_client.py
```

**Terminal 2 — Start the server (capture + detection + streaming):**
```bash
python stream_server.py                        # webcam
python stream_server.py --source video.mp4     # video file
python stream_server.py --no-display           # headless (no OpenCV window)
```

Then open **http://localhost:5050** to view the received stream.

### CLI Options

#### `app.py`
| Flag | Default | Description |
|:-----|:--------|:-----------|
| `--source` | `0` (webcam) | Video source — integer for webcam index, or file path |
| `--port` | `5050` | Web dashboard port |
| `--embedder` | `mobilenet` | `mobilenet` for DeepSORT, `none` for pure IoU |

#### `stream_server.py`
| Flag | Default | Description |
|:-----|:--------|:-----------|
| `--source` | `0` | Video source |
| `--port` | `9999` | UDP destination port |
| `--ip` | `127.0.0.1` | UDP destination IP |
| `--no-display` | `false` | Disable local OpenCV preview |
| `--embedder` | `mobilenet` | Tracker appearance embedder |

#### `stream_client.py`
| Flag | Default | Description |
|:-----|:--------|:-----------|
| `--port` | `9999` | UDP listen port |
| `--web-port` | `5050` | Web dashboard port |
| `--web-host` | `0.0.0.0` | Web bind address |



## Configuration Reference

All tunables in **`config.py`**:

### Video Input
| Parameter | Default | Description |
|:----------|:--------|:-----------|
| `INPUT_SOURCE` | `0` | Webcam index or video file path |
| `FRAME_WIDTH` | `640` | Processing width (pixels) |
| `FRAME_HEIGHT` | `480` | Processing height (pixels) |

### YOLOv8 Detection
| Parameter | Default | Description |
|:----------|:--------|:-----------|
| `YOLO_MODEL` | `yolov8n.pt` | Model variant — `yolov8n.pt` (fast) to `yolov8x.pt` (accurate) |
| `YOLO_CONF` | `0.5` | Minimum confidence threshold |
| `YOLO_IOU` | `0.7` | NMS IoU threshold |
| `YOLO_IMGSZ` | `640` | Inference resolution |
| `YOLO_DEVICE` | `""` (auto) | `""` = auto, `"cpu"`, `"cuda:0"`, `"mps"` |
| `TARGET_CLASSES` | `[0,1,2,3,5,7]` | COCO class IDs to detect |

### DeepSORT Tracking
| Parameter | Default | Description |
|:----------|:--------|:-----------|
| `TRACKER_MAX_AGE` | `30` | Frames before a lost track is deleted |
| `TRACKER_N_INIT` | `3` | Detections needed to confirm a track |
| `TRACKER_EMBEDDER` | `"mobilenet"` | `"mobilenet"` for appearance features, `None` for faster IoU-only |
| `TRAJECTORY_LENGTH` | `30` | Points per trajectory tail |

### UDP Protocol
| Parameter | Default | Description |
|:----------|:--------|:-----------|
| `UDP_IP` | `127.0.0.1` | Destination IP address |
| `UDP_PORT` | `9999` | Destination port |
| `JPEG_QUALITY` | `70` | Compression quality (0-100) |
| `MAX_DGRAM_SIZE` | `9216` | Max UDP datagram size (macOS default) |

### Visualisation
| Parameter | Default | Description |
|:----------|:--------|:-----------|
| `CLASS_COLORS` | See config | BGR colour dict per class |
| `WEB_PORT` | `5050` | Browser dashboard port |



## How It Works

### 1. Object Detection

The **`ObjectDetector`** class wraps Ultralytics YOLOv8:

```python
from detector import ObjectDetector

detector = ObjectDetector()
detections = detector.detect(frame)
# Returns: [{"bbox": [x,y,w,h], "confidence": 0.92, "class_name": "person", "class_id": 0}, ...]
```

- **Model:** YOLOv8n (nano) — 3.2M parameters, 8.7 GFLOPs
- **Pre-trained** on COCO (80 classes), filtered to 6 relevant classes
- **NMS** applied automatically to eliminate duplicate detections
- Inference resolution configurable (default 640px)

### 2. Multi-Object Tracking

The **`ObjectTracker`** class wraps DeepSORT:

```python
from tracker import ObjectTracker

tracker = ObjectTracker()
tracks = tracker.update(detections, frame)
trajectories = tracker.get_trajectories()
```

**How DeepSORT maintains IDs across frames:**

1. **Kalman Filter** predicts where each tracked object will be in the next frame
2. **Hungarian Algorithm** matches predictions to new detections using a cost matrix
3. **MobileNet Embedder** extracts appearance features to handle cases where IoU alone fails (e.g., objects crossing paths)
4. **Track Lifecycle:**
   - `n_init=3` → A new track needs 3 consecutive detections to be confirmed
   - `max_age=30` → Lost tracks survive 30 frames via Kalman prediction (handles occlusion)
   - Stale tracks are pruned automatically

**Occlusion Handling:** When an object is temporarily hidden (e.g., behind another object), the Kalman filter continues predicting its position. When the object reappears within `max_age` frames, it's re-associated with the same ID.

### 3. Annotation & Visualisation

Each frame is annotated with:
- **Coloured bounding boxes** (colour-coded by class)
- **Labels** showing `ID:{track_id} {class} {confidence}`
- **Trajectory polylines** — last 30 centre-point positions per track
- **HUD overlay** — real-time FPS and object count

### 4. UDP Streaming Protocol

Custom binary packet format for network streaming:

```
┌────────────┬──────────────┬─────────────┬──────────────┬───────────┬───────────┐
│ frame_id   │ object_count │ chunk_index │ total_chunks │ timestamp │ JPEG data │
│ 4 bytes    │ 2 bytes      │ 2 bytes     │ 2 bytes      │ 8 bytes   │ variable  │
│ uint32     │ uint16       │ uint16      │ uint16       │ float64   │           │
└────────────┴──────────────┴─────────────┴──────────────┴───────────┴───────────┘
         Header: 18 bytes                              Payload: up to 9198 bytes
```

**Fragmentation:** Frames exceeding the max datagram size are split into numbered chunks and reassembled on the client. Incomplete frames older than 500ms are discarded (graceful packet-loss handling).


## Performance

### Benchmarks (Apple M-series CPU)

| Stage | Time per Frame | Notes |
|:------|:--------------|:------|
| Frame capture + resize | ~2 ms | OpenCV VideoCapture |
| YOLOv8n inference | ~25-40 ms | 640px resolution |
| DeepSORT update (MobileNet) | ~5-15 ms | Appearance embedding |
| DeepSORT update (IoU only) | ~1-2 ms | `--embedder none` |
| Annotation drawing | ~1-2 ms | Boxes, labels, polylines |
| JPEG encoding | ~1-2 ms | Quality 80 |
| **Total (MobileNet)** | **~35-60 ms** | **~17-28 FPS** |
| **Total (IoU only)** | **~30-45 ms** | **~22-33 FPS** |

### Tuning Tips

| Want more FPS? | Do this |
|:--------------|:--------|
| Fastest easy win | `python app.py --embedder none` (skip appearance features) |
| Lower resolution | Set `YOLO_IMGSZ = 320` in config.py |
| Fewer classes | Reduce `TARGET_CLASSES` list |
| GPU acceleration | Set `YOLO_DEVICE = "cuda:0"` (NVIDIA) or `"mps"` (Apple Silicon) |
| Smaller model | Already using YOLOv8n (smallest). Goes up to `yolov8x.pt` for accuracy |

## Troubleshooting

| Problem | Solution |
|:--------|:--------|
| **"Cannot open video source"** | Check webcam permissions in System Settings → Privacy → Camera |
| **Low FPS (<10)** | Use `--embedder none` or lower `YOLO_IMGSZ` to 320 |
| **Port already in use** | Change `--port` flag or kill the process using that port |
| **Model download fails** | Manually download `yolov8n.pt` from [Ultralytics releases](https://github.com/ultralytics/assets/releases) |
| **`pkg_resources` error** | Run `pip install "setuptools<71"` (Python 3.14 compatibility) |
| **UDP packets not arriving** | Check `sysctl net.inet.udp.maxdgram` on macOS (default 9216) |
| **Black/blank video in browser** | Ensure the pipeline thread started — check terminal for `[pipeline] Running` |

