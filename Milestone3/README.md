# Milestone 3 - Real-Time Inference & Logging

# Overview
This milestone implements real-time PPE detection using YOLOv11.
The Python script processes live webcam feed, detects violations,
sends Telegram alerts, and logs all events to violation_log.json.

# Files
- PPE.py - Main detection script
- best.pt - Trained YOLOv11 model weights
- violation_log.json - Logged violation records
- predicted_result.jpg - Sample detection output
- PPE_Dashboard.pbix - Power BI dashboard file
- Terminal_Output.png - Console alert screenshot

# How It Works
1. Webcam captures live frames
2. YOLOv11 detects PPE violations in each frame
3. Telegram alert sent with snapshot if violation found
4. Violation logged to JSON file
5. Power BI dashboard reads JSON for visualization

# Tech Stack
- Python, OpenCV, Ultralytics YOLOv11
- Telegram Bot API
- Power BI
