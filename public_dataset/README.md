# Smart Traffic Control System

YOLO-based vehicle detection with ESP32-CAM-controlled traffic lights.

## Implemented
- ESP32-CAM live video streaming.
- Custom-trained YOLO vehicle detection.
- Python/OpenCV inference.
- Serial communication with ESP32-CAM.
- Red, yellow and green LED control.

## Traffic Logic
- 0-1 detected vehicles: Green
- 2 detected vehicles: Yellow
- 3 or more detected vehicles: Red

This is a prototype demonstration rule, not a validated real-world traffic control algorithm.

## Hardware
- AI Thinker ESP32-CAM
- Red, yellow and green LED module
- Windows PC with Python and compatible NVIDIA GPU (optional)

## Wiring
See docs/hardware_wiring.md.

## Setup

Install dependencies:

    python -m pip install -r requirements.txt

Place the trained model at:

    models/best.pt

Update the camera IP, serial COM port and model path in python/esp32_yolo_live.py.

Run:

    python python/esp32_yolo_live.py

## Limitations
Vehicle count is based on detections in individual frames, not persistent vehicle tracking. Accident detection and emergency alerts are not yet implemented.

## Future Work
- Vehicle tracking and lane-wise counting.
- Adaptive traffic signal timing.
- Accident-event detection.
- Emergency vehicle prioritization.
- Dashboard and performance reporting.
