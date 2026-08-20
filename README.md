# VisionAI — Age Detection & Hand Gesture Recognition 🧠✋

An ultra-fast, real-time Computer Vision desktop application built with Python, OpenCV, CustomTkinter, and MediaPipe. Features age estimation and hand gesture recognition with zero frame lag and low-light support.

---

## ✨ Features

- **🧠 Age Detector:**
  - Multi-cascade face detection using OpenCV with CLAHE enhancement for dark environments.
  - Geometry and texture-based age estimation.
  - Multi-face and full-body image support.

- **✋ Gesture Recognizer:**
  - Real-time hand landmark tracking powered by MediaPipe.
  - Built-in OpenCV skin-color contour fallback if MediaPipe is unavailable.
  - Detects 11+ gestures: Thumbs Up, Thumbs Down, Peace, OK, Rock, Pointing, Open Hand, Fist, Call Me, I Love You, and Two Fingers.

- **⚡ Performance Pipeline:**
  - 3-Thread architecture (Capture, Process, GUI Dispatch) for smooth UI and zero freeze during camera stream.

---

## 🛠️ Installation

1. **Clone the repository:**
    ```bash
   git clone [https://github.com/itsrameen/vision-bridge-ai.git](https://github.com/itsrameen/vision-bridge-ai.git)
   cd vision-bridge-ai

2. **Install dependencies:**
   ```
   pip install customtkinter opencv-python pillow mediapipe numpy

---

## 🚀 Usage
Run the main application script:
```
 python vision.py
```

## 💻 Tech Stack

- **UI Framework:** CustomTkinter
- **Computer Vision:** OpenCV (cv2) & MediaPipe
- **Image Processing:** PIL (Pillow) & NumPy
