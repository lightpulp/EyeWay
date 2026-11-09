# EyeWay

*A Low-Cost, Haptic-Feedback Mobility Aid for Blind & Visually Impaired Users*

EyeWay is a two-part wearable system. Lightweight glasses handle camera streaming and haptic feedback, while a smartphone runs AI-based obstacle detection. The user receives directional tactile cues through vibration motors embedded along the glasses strap.

## 👓 Hardware: Glasses Unit

The glasses unit is designed to be lightweight, low-power, and discreet. It attaches to standard eyeglass frames using the existing strap mechanism.

### Component List

| Component | Specification | Quantity | Function |
|:---|:---|:---:|:---|
| ESP32-CAM | OV2640 camera, WiFi, GPIO pins | 1 | Captures video, streams to phone, controls motors |
| Coin Vibration Motors | 10mm x 3mm, 3V, 60-90mA | 4 | Provide directional haptic feedback |

## 📱 Software: Detection Unit

The smartphone (or a PC during development) runs a MobileNetV2 model that classifies obstacles from the camera stream. Detected obstacles trigger alerts that the glasses translate into haptic cues.

**Detected classes:** chair, door, fence, garbage bin, obstacle, plant, pothole, stairs, table, vehicle.

## 🏗️ System Architecture

```
ESP32-CAM ──WiFi──► Detection Server (Flask + MobileNetV2) ──► Alert Logic ──WiFi──► ESP32 Motors
```

1. The ESP32-CAM captures frames and serves them over HTTP.
2. The detection server classifies each frame and returns the class and confidence.
3. If confidence exceeds the threshold (0.8), the alert logic picks a direction.
4. The ESP32 drives the matching vibration motors.

## 🔊 Feedback Mapping

| Obstacle Position | Motors Activated | Pattern |
|:---|:---|:---|
| Left | Left motor | Short pulse |
| Right | Right motor | Short pulse |
| Center | Both motors | Continuous pulse |
| Close range (large in frame) | All motors | Rapid pulse |

Direction is estimated from where the obstacle sits in the frame (left, center, or right third).

## ⚙️ Getting Started

### Prerequisites

- Python 3.8+ with pip
- Arduino IDE with the ESP32 board package
- ESP32-CAM flashed with the CameraWebServer sketch

### Install Python Dependencies

```bash
pip install flask opencv-python requests torch torchvision pillow numpy
```

### Run the Detection Server

```bash
cd api
python main.py
```

The server listens on `http://127.0.0.1:5000`.

### Run the Live Interface

```bash
python live_ui.py
```

Open **http://127.0.0.1:8000** to view the live feed with predictions and alerts.

## 🙏 Acknowledgements

EyeWay builds on the research and dataset from the original [Obstacle Detection with Alert System for Visually Impaired](https://github.com/drisskhattabi6/Obstacle-Detection-with-Alert-System-for-Visually-Impaired) project, which uses a CNN and fine-tuned MobileNetV2 with Node-RED for audio alerts.