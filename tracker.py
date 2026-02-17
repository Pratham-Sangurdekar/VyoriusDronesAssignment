"""
Multi-Object Tracker – wraps DeepSORT for persistent track IDs & trajectories.

Usage:
    from tracker import ObjectTracker
    trk = ObjectTracker()
    tracks = trk.update(detections, frame)
    trajectories = trk.get_trajectories()
"""

from collections import deque

from deep_sort_realtime.deepsort_tracker import DeepSort

import config


class ObjectTracker:
    """Manages DeepSORT tracker + per-track trajectory history."""

    def __init__(
        self,
        max_age: int = config.TRACKER_MAX_AGE,
        n_init: int = config.TRACKER_N_INIT,
        max_iou_distance: float = config.TRACKER_MAX_IOU_DISTANCE,
        max_cosine_distance: float = config.TRACKER_MAX_COSINE_DISTANCE,
        nn_budget: int = config.TRACKER_NN_BUDGET,
        embedder: str | None = config.TRACKER_EMBEDDER,
        trajectory_length: int = config.TRAJECTORY_LENGTH,
    ):
        self.max_age = max_age
        self.trajectory_length = trajectory_length

        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder=embedder,
            half=True,
        )

        # track_id -> deque of (cx, cy) centre points
        self._trajectories: dict[int, deque] = {}
        # track_id -> frames since last seen (for pruning)
        self._last_seen: dict[int, int] = {}
        self._frame_count = 0

    # ── public API ───────────────────────────────────────────────────────
    def update(self, detections: list[dict], frame) -> list[dict]:
        """Feed new detections and get back confirmed tracks.

        Parameters
        ----------
        detections : list[dict]
            Each dict must have keys: bbox ([x,y,w,h]), confidence, class_name
        frame : np.ndarray
            Current BGR frame (needed by appearance embedder).

        Returns
        -------
        list[dict]  with keys: track_id, bbox_ltrb, class_name, confidence
        """
        self._frame_count += 1

        # Convert to DeepSORT input: list of ([x,y,w,h], conf, class_name)
        ds_detections = [
            (d["bbox"], d["confidence"], d["class_name"]) for d in detections
        ]

        raw_tracks = self._tracker.update_tracks(
            ds_detections, frame=frame
        )

        active_ids: set[int] = set()
        results: list[dict] = []

        for track in raw_tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            active_ids.add(tid)
            l, t, r, b = track.to_ltrb()
            cx, cy = int((l + r) / 2), int((t + b) / 2)

            # Update trajectory
            if tid not in self._trajectories:
                self._trajectories[tid] = deque(maxlen=self.trajectory_length)
            self._trajectories[tid].append((cx, cy))
            self._last_seen[tid] = self._frame_count

            results.append(
                {
                    "track_id": tid,
                    "bbox_ltrb": [int(l), int(t), int(r), int(b)],
                    "class_name": track.get_det_class(),
                    "confidence": track.get_det_conf(),
                }
            )

        # Prune stale trajectories
        stale = [
            tid
            for tid, last in self._last_seen.items()
            if (self._frame_count - last) > self.max_age
        ]
        for tid in stale:
            self._trajectories.pop(tid, None)
            self._last_seen.pop(tid, None)

        return results

    def get_trajectories(self) -> dict[int, list[tuple[int, int]]]:
        """Return {track_id: [(cx, cy), ...]} for drawing polylines."""
        return {tid: list(pts) for tid, pts in self._trajectories.items()}
