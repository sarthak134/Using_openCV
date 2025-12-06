# detect_yolo.py (Optimized for full-frame pipeline)
import cv2
from ultralytics import YOLO

_model = None

def get_yolo_model(model_path: str = "yolov8n.pt"):
    """
    Loads YOLO model only once (global cache).
    Fastest method for long videos.
    """
    global _model
    if _model is None:
        _model = YOLO(model_path)  # loads or downloads
    return _model


def detect_objects_with_boxes(frame, model=None, conf=0.25):
    """
    Fast YOLO detection.
    Returns list of:
        { "label": str, "box": [x1, y1, x2, y2] }
    Used directly by pipeline + centroid tracker.
    """

    if model is None:
        model = get_yolo_model()

    # Ultralytics 8 fastest inference mode
    results = model.predict(frame, conf=conf, verbose=False)[0]

    detections = []
    h, w = frame.shape[:2]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names.get(cls_id, str(cls_id))

        x1, y1, x2, y2 = box.xyxy[0].tolist()  # already float

        # convert to int + clamp
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))

        detections.append({
            "label": label,
            "box": [x1, y1, x2, y2]
        })

    return detections
