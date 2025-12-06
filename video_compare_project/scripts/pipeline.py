# pipeline.py (Frame-by-Frame + Live Preview + Full Analysis)
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

from detect_yolo import detect_objects_with_boxes, get_yolo_model
from motion_farneback import compute_farneback_motion
from centroid_tracker import CentroidTracker

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

VIDEO_1 = os.path.join(PROJECT_ROOT, "input_videos", "video1.mp4")
VIDEO_2 = os.path.join(PROJECT_ROOT, "input_videos", "video2.mp4")

OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "outputs", "comparison_full.mp4")
REPORT_JSON = os.path.join(PROJECT_ROOT, "outputs", "analysis_report.json")
REPORT_PDF = os.path.join(PROJECT_ROOT, "outputs", "analysis_report.pdf")
GRAPH_DIR = os.path.join(PROJECT_ROOT, "outputs", "graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

FRAME_W, FRAME_H = 640, 480
SSIM_DOWNSCALE = 0.5  
YOLO_CONF = 0.25

# Load YOLO model once
yolo = get_yolo_model()
print("YOLO model loaded.\n")

# Trackers
trackers_v1 = {}
trackers_v2 = {}

unique_counts_v1 = {}
unique_counts_v2 = {}

# Graph data
ssim_series = []
motion_series = []
frame_indices = []


# --------------------------------------------------------
# PDF Helper
# --------------------------------------------------------
def embed_image_to_pdf(c, img_path, x, y, w):
    try:
        im = Image.open(img_path)
        iw, ih = im.size
        aspect = ih / iw
        h = int(w * aspect)
        c.drawImage(img_path, x, y - h, width=w, height=h)
        return h
    except:
        return 0


def generate_pdf(summary, table, pdf_path, graph_paths):

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 40

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, "Video Comparison Analysis (Frame-by-Frame)")
    y -= 30

    # Summary
    c.setFont("Helvetica", 11)
    for k, v in summary.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 15

    # --------------------------------------------------------
    # STRUCTURED CLEAN TABLE (NEW)
    # --------------------------------------------------------
    y -= 15
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "YOLO Class Summary")
    y -= 22

    # Header row
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40,  y,   "Class")
    c.drawString(170, y,   "V1 Count")
    c.drawString(260, y,   "V2 Count")
    c.drawString(350, y,   "Missing")
    c.drawString(440, y,   "New")
    y -= 14

    # Separator line
    c.setLineWidth(0.6)
    c.line(40, y, 520, y)
    y -= 12

    # Table rows
    c.setFont("Helvetica", 10)

    for row in table:
        if y < 120:
            c.showPage()
            y = height - 60

            # Page header
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, y, "YOLO Class Summary (cont.)")
            y -= 20

            c.setFont("Helvetica-Bold", 11)
            c.drawString(40,  y,   "Class")
            c.drawString(170, y,   "V1 Count")
            c.drawString(260, y,   "V2 Count")
            c.drawString(350, y,   "Missing")
            c.drawString(440, y,   "New")
            y -= 14

            c.line(40, y, 520, y)
            y -= 12
            c.setFont("Helvetica", 10)

        # Row
        c.drawString(40,  y, row["class"])
        c.drawString(170, y, str(row["v1"]))
        c.drawString(260, y, str(row["v2"]))
        c.drawString(350, y, str(row["missing"]))
        c.drawString(440, y, str(row["new"]))
        y -= 14

    # --------------------------------------------------------
    # GRAPHS PAGES
    # --------------------------------------------------------
    for gp in graph_paths:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, height - 50, os.path.basename(gp))
        embed_image_to_pdf(c, gp, 40, height - 80, w=520)

    c.save()


# --------------------------------------------------------
# Tracker update per frame
# --------------------------------------------------------
def update_trackers_for_frame(detections, trackers_dict):

    per_class_boxes = {}
    for det in detections:
        label = det["label"]
        per_class_boxes.setdefault(label, []).append(det["box"])

    per_class_counts = {}

    for label, boxes in per_class_boxes.items():
        if label not in trackers_dict:
            trackers_dict[label] = CentroidTracker(max_disappeared=12, max_distance=60)

        tr = trackers_dict[label]
        tr.update(boxes)
        per_class_counts[label] = len(boxes)

    for label in trackers_dict.keys():
        per_class_counts.setdefault(label, 0)

    return per_class_counts


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():

    cap1 = cv2.VideoCapture(VIDEO_1)
    cap2 = cv2.VideoCapture(VIDEO_2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("ERROR: cannot open input videos.")
        return

    fps = int(cap1.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    out_writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (FRAME_W * 2, FRAME_H)
    )

    frame_idx = 0
    prev_gray = None
    analysis = []

    print("\nProcessing ALL frames...\n")

    while True:

        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 or not ret2:
            break

        f1s = cv2.resize(f1, (FRAME_W, FRAME_H))
        f2s = cv2.resize(f2, (FRAME_W, FRAME_H))

        # SSIM
        g1_small = cv2.cvtColor(
            cv2.resize(f1s, (int(FRAME_W * SSIM_DOWNSCALE), int(FRAME_H * SSIM_DOWNSCALE))),
            cv2.COLOR_BGR2GRAY
        )
        g2_small = cv2.cvtColor(
            cv2.resize(f2s, (int(FRAME_W * SSIM_DOWNSCALE), int(FRAME_H * SSIM_DOWNSCALE))),
            cv2.COLOR_BGR2GRAY
        )
        ssim_score, _ = ssim(g1_small, g2_small, full=True)

        # Motion
        g1_gray = cv2.cvtColor(f1s, cv2.COLOR_BGR2GRAY)
        motion_val = 0.0 if prev_gray is None else compute_farneback_motion(prev_gray, g1_gray)
        prev_gray = g1_gray.copy()

        # YOLO detection (every frame)
        dets1 = detect_objects_with_boxes(f1s, model=yolo, conf=YOLO_CONF)
        dets2 = detect_objects_with_boxes(f2s, model=yolo, conf=YOLO_CONF)

        counts1 = update_trackers_for_frame(dets1, trackers_v1)
        counts2 = update_trackers_for_frame(dets2, trackers_v2)

        # Unique counts
        for lbl, tr in trackers_v1.items():
            unique_counts_v1[lbl] = tr.unique_count()
        for lbl, tr in trackers_v2.items():
            unique_counts_v2[lbl] = tr.unique_count()

        analysis.append({
            "frame": frame_idx,
            "ssim": float(ssim_score),
            "motion": float(motion_val),
            "objects_video1": counts1,
            "objects_video2": counts2
        })

        ssim_series.append(float(ssim_score))
        motion_series.append(float(motion_val))
        frame_indices.append(frame_idx)

        combined = np.hstack((f1s, f2s))

        # Live window
        cv2.putText(
            combined,
            f"Frame {frame_idx} | SSIM: {ssim_score:.3f} | MOT: {motion_val:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1
        )

        cv2.imshow("Video Comparison (Press Q to Quit)", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out_writer.write(combined)

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_frames}")

        frame_idx += 1

    cap1.release()
    cap2.release()
    out_writer.release()
    cv2.destroyAllWindows()

    # SAVE JSON
    with open(REPORT_JSON, "w") as f:
        json.dump(analysis, f, indent=2)

    # CLASS SUMMARY
    all_classes = set(list(unique_counts_v1.keys()) + list(unique_counts_v2.keys()))
    table = []

    for cls in sorted(all_classes):
        v1c = unique_counts_v1.get(cls, 0)
        v2c = unique_counts_v2.get(cls, 0)
        table.append({
            "class": cls,
            "v1": int(v1c),
            "v2": int(v2c),
            "missing": max(0, v1c - v2c),
            "new": max(0, v2c - v1c)
        })

    # GRAPHS
    gp_ssim = os.path.join(GRAPH_DIR, "ssim_vs_frame.png")
    gp_motion = os.path.join(GRAPH_DIR, "motion_vs_frame.png")
    gp_counts = os.path.join(GRAPH_DIR, "object_counts.png")

    # SSIM
    plt.figure(figsize=(6.5, 3.5))
    plt.plot(frame_indices, ssim_series)
    plt.xlabel("Frame")
    plt.ylabel("SSIM")
    plt.title("SSIM vs Frame")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(gp_ssim, dpi=150)
    plt.close()

    # Motion
    plt.figure(figsize=(6.5, 3.5))
    plt.plot(frame_indices, motion_series, color="orange")
    plt.xlabel("Frame")
    plt.ylabel("Motion")
    plt.title("Motion vs Frame")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(gp_motion, dpi=150)
    plt.close()

    # Unique counts
    classes_sorted = [r["class"] for r in table]
    v1_vals = [r["v1"] for r in table]
    v2_vals = [r["v2"] for r in table]

    plt.figure(figsize=(6.5, 3.5))
    x = np.arange(len(classes_sorted))
    width = 0.35
    plt.bar(x - width/2, v1_vals, width, label="Video1")
    plt.bar(x + width/2, v2_vals, width, label="Video2")
    plt.xticks(x, classes_sorted, rotation=45, ha='right')
    plt.ylabel("Unique object count")
    plt.title("Unique Object Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(gp_counts, dpi=150)
    plt.close()

    # PDF
    summary = {
        "Total frames": frame_idx,
        "Frames processed": frame_idx,
        "Average SSIM": round(float(np.mean(ssim_series)), 4),
        "Average Motion": round(float(np.mean(motion_series)), 4),
        "Analysis time": datetime.now().isoformat()
    }

    generate_pdf(summary, table, REPORT_PDF, [gp_ssim, gp_motion, gp_counts])

    print("\n---- DONE ----")
    print("JSON:", REPORT_JSON)
    print("PDF :", REPORT_PDF)
    print("Video:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
