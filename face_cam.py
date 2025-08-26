# face_cam.py
# Minimal YOLOv8-Face webcam script (PyTorch backend).
# Put your face weights at: weights/yolov8n-face.pt  (or change WEIGHTS below)

import os
import sys
from ultralytics import YOLO

WEIGHTS = r"weights\yolov8n-face.pt"   # <-- change if your filename/path is different

def main():
    if not os.path.exists(WEIGHTS):
        print(f"[!] Face weights not found: {WEIGHTS}\n"
              f"    Put your .pt file there or update WEIGHTS.")
        sys.exit(1)

    model = YOLO(WEIGHTS)  # loads face detector
    # Webcam: source=0; start simple on CPU first
    model.predict(
        source=0,
        conf=0.25,
        imgsz=640,
        device="cpu",   # keep "cpu" for now; we can switch to Intel iGPU (OpenVINO) next step
        show=True
    )

if __name__ == "__main__":
    main()
