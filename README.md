# AR-Camera-Calibration-and-Augmented-Reality

This project bridges the physical and digital worlds by combining **camera calibration**, **pose estimation**, and **augmented reality** using OpenCV and C++. A live camera feed is used to detect a checkerboard (or a custom target), calibrate the camera, and render virtual 3D objects aligned with the real-world scene.

> Developed by: **Atharva Nayak** & **Akshaj Raut**

---

## 🚀 Features

- ✅ Detect checkerboard patterns in live video
- ✅ Calibrate intrinsic camera parameters using OpenCV
- ✅ Estimate camera pose with `solvePnP`
- ✅ Project 3D axes and corners using `projectPoints`
- ✅ Render a virtual 3D car in real-world coordinates
- ✅ ORB feature detection for custom targets (e.g., membership card)
- ✅ Real-time video processing with live object tracking

---

## 📂 Project Structure

```bash
.
├── src/
│   ├── main.cpp         # Entry point: video capture & UI
│   ├── orb.cpp          # ORB feature detection
│   ├── pose.cpp         # Pose estimation and projection
│   ├── extension.cpp    # Virtual object (car) rendering
│   └── read_obj.cpp     # Optional 3D object loader
├── CMakeLists.txt       # Build configuration
├── metadata             # Calibration parameters (YAML)
├── Project_4_Report.pdf # Full technical documentation
├── .gitignore           # Ignore build/artifact files
└── README.md            # You're here!
```
##  🔧 Installation & Running
### 🔨 Build Instructions

```bash
# Clone the repository
git clone https://github.com/atharvanayak25/AR-Camera-Calibration-and-Augmented-Reality.git
cd AR-Camera-Calibration-and-Augmented-Reality

# Create a build directory
mkdir build && cd build

# Run CMake and build
cmake ..
make

```
### ▶️ Run the Program

```bash
./AR_Camera_Calibration_and_Augmented_Reality
```

### 🧪 Controls & Interactions
s — Save calibration frame (checkerboard detected)

c — Calibrate camera (after at least 5 saved frames)

w — Save calibration parameters to file

ESC — Exit the program

### 🧠 Concepts Used
Camera Calibration: Determining intrinsic matrix and distortion coefficients

Pose Estimation: Using 2D-3D correspondences with solvePnP

Augmented Reality: Rendering a 3D object (car) with real-time pose tracking

ORB Feature Matching: Robust feature detection on non-checkerboard targets

OpenCV + C++: Image processing, computer vision, and rendering

### 📽️ Demo
Video: https://drive.google.com/drive/folders/1X0oZIE2Mi2Tef0i6TXigVjsIMT2ENV28?usp=sharing

### 🙌 Acknowledgements
OpenCV for providing powerful computer vision tools
ORB Feature Detection research and community resources
