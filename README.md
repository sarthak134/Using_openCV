# Using_openCV
A full video comparison system that analyzes two videos frame-by-frame using YOLO object detection, SSIM similarity, and optical-flow motion analysis. It tracks objects, counts differences, generates comparison videos, graphs, JSON data, and a detailed PDF report for complete visual analysis.
# 🎥 Video Comparison & Analysis System (YOLO + SSIM + Motion + PDF Report)

This project performs a complete **frame-by-frame comparison** of two videos using advanced computer vision techniques.  
It detects visual differences, counts objects, measures motion, calculates structural similarity, and generates a detailed PDF report with graphs.

---

## 🚀 Features

- **Frame-by-frame processing** of two videos  
- **YOLO-based object detection** with per-class unique counting  
- **Centroid tracking** to track objects across frames  
- **SSIM (Structural Similarity Index)** comparison  
- **Farneback optical flow** motion analysis  
- **Combined output video** showing both inputs side-by-side  
- **Live preview during processing**  
- **JSON analysis file** for every frame  
- **PDF report** with:
  - Summary  
  - Structured object-count table  
  - SSIM graph  
  - Motion graph  
  - Object count graph  

---

## 📁 Folder Structure

project/
│── input_videos/
│── outputs/
│ ├── comparison_full.mp4
│ ├── analysis_report.json
│ ├── analysis_report.pdf
│ └── graphs/
│── pipeline.py
│── detect_yolo.py
│── motion_farneback.py
│── centroid_tracker.py
│── requirements.txt



---

## 🔧 Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt


python pipeline.py


| File                     | Description                           |
| ------------------------ | ------------------------------------- |
| **comparison_full.mp4**  | Side-by-side comparison video         |
| **analysis_report.json** | Frame-level metrics                   |
| **analysis_report.pdf**  | Full summary + tables + graphs        |
| **graphs/**              | SSIM, motion, and object count charts |



🧠 Technologies Used

Python

OpenCV

YOLO (Ultralytics)

SSIM (skimage)

Farneback Optical Flow

ReportLab

NumPy / Matplotlib



---

# 🎉 Your README is now ready for GitHub.

If you want, I can also add:

✔ Badges (Python version, License, Stars)  
✔ A project banner  
✔ Example output screenshots  
✔ Demo video GIF  
✔ A professional description section  


