# centroid_tracker.py (Optimized for full-frame + YOLO-every-2 pipeline)
import numpy as np
from scipy.spatial import distance


class CentroidTracker:
    """
    Optimized centroid tracker for per-class object tracking.
    Works with YOLO detections that may appear every few frames.
    """

    def __init__(self, max_disappeared=12, max_distance=60):
        # YOLO may skip multiple frames → increase tolerance
        self.next_object_id = 0
        self.objects = {}          # id -> (cX, cY)
        self.disappeared = {}      # id -> disappeared counter
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # track unique object count
        self.counted_ids = set()

    # ----------------------------------------------------
    def register(self, centroid):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.counted_ids.add(oid)
        self.next_object_id += 1
        return oid

    # ----------------------------------------------------
    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
        if oid in self.disappeared:
            del self.disappeared[oid]

    # ----------------------------------------------------
    def update(self, rects):
        """
        rects: list of bounding boxes → [x1, y1, x2, y2]
        Returns: dict of active tracked objects → {id: (cX, cY)}
        """

        # compute centroids for new detections
        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            cX = int((x1 + x2) * 0.5)
            cY = int((y1 + y2) * 0.5)
            input_centroids.append((cX, cY))

        # CASE 1: No objects currently being tracked
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects.copy()

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # CASE 2: No detections in this frame
        if len(input_centroids) == 0:
            for oid in object_ids:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects.copy()

        # ---------------------------------------------
        # Compute distance matrix between old + new centroids
        # ---------------------------------------------
        D = distance.cdist(np.array(object_centroids), np.array(input_centroids))

        # SMALL OPTIMIZATION: sort matching by smallest distances first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        # ---------------------------------------------
        # Match existing objects with new detections
        # ---------------------------------------------
        for r, c in zip(rows, cols):

            if r in used_rows or c in used_cols:
                continue

            if D[r, c] > self.max_distance:
                continue

            oid = object_ids[r]
            self.objects[oid] = input_centroids[c]
            self.disappeared[oid] = 0

            used_rows.add(r)
            used_cols.add(c)

        # ---------------------------------------------
        # Register NEW detections not matched
        # ---------------------------------------------
        for i, centroid in enumerate(input_centroids):
            if i not in used_cols:
                self.register(centroid)

        # ---------------------------------------------
        # Handle objects that disappeared
        # ---------------------------------------------
        for i, oid in enumerate(object_ids):
            if i not in used_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        return self.objects.copy()

    # ----------------------------------------------------
    def unique_count(self):
        """Return number of unique tracked object IDs."""
        return len(self.counted_ids)
