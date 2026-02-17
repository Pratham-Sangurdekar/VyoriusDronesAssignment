"""
Object Detector – wraps Ultralytics YOLOv8 for real-time inference.

Usage:
    from detector import ObjectDetector
    det = ObjectDetector()
    detections = det.detect(frame)   # frame: BGR numpy array
"""

from ultralytics import YOLO
import config


class ObjectDetector:
    """Thin wrapper around YOLOv8 that returns detections in a tracker-friendly format."""

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL,
        conf: float = config.YOLO_CONF,
        iou: float = config.YOLO_IOU,
        imgsz: int = config.YOLO_IMGSZ,
        device: str = config.YOLO_DEVICE,
        target_classes: list[int] | None = None,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device if device else None  # None = auto-select
        self.target_classes = target_classes if target_classes is not None else config.TARGET_CLASSES

    # ── public API ───────────────────────────────────────────────────────
    def detect(self, frame) -> list[dict]:
        """Run inference on a single BGR frame.

        Returns a list of dicts, each with keys:
            bbox        – [x, y, w, h]  (left, top, width, height)
            confidence  – float 0-1
            class_name  – str, e.g. "person"
            class_id    – int (COCO id)
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.target_classes,
            verbose=False,
        )

        detections: list[dict] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            detections.append(
                {
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # [left, top, w, h]
                    "confidence": round(box.conf[0].item(), 3),
                    "class_name": self.model.names[cls_id],
                    "class_id": cls_id,
                }
            )
        return detections
