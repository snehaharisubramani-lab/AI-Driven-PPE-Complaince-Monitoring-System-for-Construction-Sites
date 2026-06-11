import os
import json
import requests
import cv2
from datetime import datetime
from ultralytics import YOLO

# ─────────────────────────────────────────────
# TELEGRAM CONFIGURATION
# ─────────────────────────────────────────────
TELEGRAM_CONFIG = {
    "enabled": True,
    "bot_token": "8980234682:AAFWgRBTUx7-ecEcLSIC0xVVX7SMYKN-Xyo",
    "chat_id":   "5002998067",
}

# ─────────────────────────────────────────────
# ALERT CONFIGURATION
# ─────────────────────────────────────────────
ALERT_CONFIG = {
    "log": {
        "enabled": True,
        "path": "violation_log.json",
    },
    "console": True,
}

MODEL_PATH = r"C:\Users\Sneha H S\Desktop\PPE PROJECT\best_v2.pt"
CAMERA_SOURCE = 0
ALERT_COOLDOWN = 30
CONFIDENCE = 0.35

# ─────────────────────────────────────────────
# PPE CLASSES
# ─────────────────────────────────────────────
required_ppe      = {"helmet", "vest", "gloves", "goggles", "boots"}
violation_classes = {"no_helmet", "no_vest", "no_gloves", "no_goggles", "no_boots"}

# ─────────────────────────────────────────────
# TELEGRAM ALERT FUNCTION
# ─────────────────────────────────────────────
def send_telegram_alert(message: str, image_path: str = None):
    cfg = TELEGRAM_CONFIG
    if not cfg["enabled"]:
        return
    try:
        url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
        payload = {
            "chat_id": cfg["chat_id"],
            "text": message,
            "parse_mode": "HTML",
        }
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            print("📱 Telegram alert sent successfully!")
        else:
            print(f"⚠️  Telegram failed: {response.text}")

        if image_path and os.path.exists(image_path):
            url2 = f"https://api.telegram.org/bot{cfg['bot_token']}/sendPhoto"
            with open(image_path, 'rb') as img:
                requests.post(url2,
                            data={"chat_id": cfg["chat_id"]},
                            files={"photo": img},
                            timeout=10)
            print("📸 Snapshot sent to Telegram!")

    except Exception as e:
        print(f"⚠️  Telegram error: {e}")


# ─────────────────────────────────────────────
# LOG FUNCTION
# ─────────────────────────────────────────────
def log_violation(image_source: str, violations: list, missing: list, status: str):
    cfg = ALERT_CONFIG["log"]
    if not cfg["enabled"]:
        return
    record = {
        "timestamp":   datetime.now().isoformat(),
        "image":       image_source,
        "status":      status,
        "violations":  violations,
        "missing_ppe": missing,
    }
    log_path = cfg["path"]
    existing = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.append(record)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"📝 Violation logged → {log_path}")


# ─────────────────────────────────────────────
# TRIGGER ALERTS
# ─────────────────────────────────────────────
def trigger_alerts(violations: list, missing: list, status: str, snapshot_path: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if ALERT_CONFIG["console"]:
        print("\n" + "!" * 50)
        print(f"🚨 PPE VIOLATION ALERT")
        print(f"Time:   {timestamp}")
        print(f"Status: {status}")
        if violations:
            print(f"Violations:  {', '.join(v.upper() for v in violations)}")
        if missing:
            print(f"Missing PPE: {', '.join(missing)}")
        print("!" * 50)

    if status == "NON-COMPLIANT":
        message = (
            f"🚨 <b>PPE VIOLATION DETECTED</b>\n\n"
            f"🕐 Time: {timestamp}\n"
            f"📷 Camera: Live CCTV\n"
            f"❌ Status: {status}\n"
            f"🔴 Violations: {', '.join(v.upper() for v in violations)}\n\n"
            f"⚠️ Immediate action required!"
        )
    else:
        message = (
            f"⚠️ <b>INCOMPLETE PPE DETECTED</b>\n\n"
            f"🕐 Time: {timestamp}\n"
            f"📷 Camera: Live CCTV\n"
            f"❌ Status: {status}\n"
            f"🧤 Missing: {', '.join(missing)}\n\n"
            f"Please ensure full PPE compliance."
        )

    send_telegram_alert(message, snapshot_path)
    log_violation("live_cctv", violations, missing, status)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("✅ Model loaded!")

# ─────────────────────────────────────────────
# LIVE DETECTION LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit()

print("🎥 Live PPE Detection Started... Press Q to quit")

last_alert_time = None
snapshot_dir = "snapshots"
os.makedirs(snapshot_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame!")
        break

    results = model.predict(frame, conf=CONFIDENCE, verbose=False)

    detected = []
    for r in results:
        for box in r.boxes:
            cls_id     = int(box.cls)
            conf       = float(box.conf)
            class_name = model.names[cls_id]
            detected.append(class_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 0, 255) if class_name in violation_classes else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    unique_detected  = set(detected)
    violations_found = [cls for cls in unique_detected if cls in violation_classes]
    missing_ppe      = [ppe for ppe in required_ppe if ppe not in unique_detected]

    now = datetime.now()
    if violations_found:
        cv2.putText(frame, "NON-COMPLIANT!", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if last_alert_time is None or (now - last_alert_time).seconds > ALERT_COOLDOWN:
            snapshot_path = os.path.join(snapshot_dir,
                           f"violation_{now.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(snapshot_path, frame)
            trigger_alerts(violations_found, [], "NON-COMPLIANT", snapshot_path)
            last_alert_time = now

    elif missing_ppe:
        cv2.putText(frame, "INCOMPLETE PPE", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

        if last_alert_time is None or (now - last_alert_time).seconds > ALERT_COOLDOWN:
            snapshot_path = os.path.join(snapshot_dir,
                           f"incomplete_{now.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(snapshot_path, frame)
            trigger_alerts([], missing_ppe, "INCOMPLETE PPE", snapshot_path)
            last_alert_time = now

    else:
        cv2.putText(frame, "COMPLIANT!", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('PPE Live Detection - Press Q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection stopped. Log saved!")
