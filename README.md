# 📌 Object Detection Using YOLO

This project utilizes the **YOLO (You Only Look Once)** model for real-time object detection. YOLO is a fast and accurate deep learning-based approach that detects objects in images and videos.

---

## ⚡ Features

- Fast object detection using the **YOLOv3** model.
- Pre-trained model for detecting multiple classes.
- Custom dataset support (modify config files as needed).
- Python-based implementation.

---

## 🔧 Requirements

Ensure you have the following dependencies installed:

```bash
pip install opencv-python numpy torch torchvision
```
---

## 📜 YOLO Model & Weights
Due to file size limitations, the YOLO weights were not pushed to the repository. You can download them from the following link: https://sourceforge.net/projects/yolov3.mirror/files/v8/yolov3.weights/download.
Once downloaded, place the yolov3.weights file in the project directory.

---

## 📝 Files in this Repository
  - Main.py – The main script for running object detection.
  - yolov3.cfg.txt – YOLOv3 configuration file.
  - yolov3.weights – Pre-trained weights file (download separately).
  - coco.names.txt – Class labels for YOLO object detection.
  - .idea/ – IDE settings (can be ignored).
  - README.md – This documentation file.

---

## ✨ Future Improvements
  - Train a custom YOLO model on a new dataset.
  - Optimize inference for real-time deployment.
  - Experiment with YOLOv5 for improved accuracy.

---

## 📌 Author
This project was created by iArudra for object detection tasks.
