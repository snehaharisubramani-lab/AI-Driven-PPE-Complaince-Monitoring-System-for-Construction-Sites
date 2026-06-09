import os
import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO

# ─────────────────────────────────────────────
# ALERT CONFIGURATION
# ─────────────────────────────────────────────
ALERT_CONFIG = {
    "email": {
        "enabled": False,          # Set True only after you add App Password
        "sender":   "your_email@gmail.com",
        "password": "your_app_password",
        "receiver": "supervisor@company.com",
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
    },
    "log": {
        "enabled": True,
        "path": "violation_log.json",
    },
    "console": True,
}

# ─────────────────────────────────────────────
# PATHS — update these if needed
# ─────────────────────────────────────────────
MODEL_PATH = r"C:\Users\Sneha H S\Desktop\PPE PROJECT\best.pt"
IMAGE_PATH = r"C:\Users\Sneha H S\Desktop\PPE forVS\predicted_result.jpg"

# ─────────────────────────────────────────────
# PPE CLASSES
# ─────────────────────────────────────────────
required_ppe = {"helmet", "vest", "gloves", "goggles", "boots"}
violation_classes = {"no_helmet", "no_vest", "no_gloves", "no_goggles", "no_boots"}

# ─────────────────────────────────────────────
# ALERT FUNCTIONS
# ─────────────────────────────────────────────

def send_email_alert(subject: str, body: str):
    cfg = ALERT_CONFIG["email"]
    if not cfg["enabled"]:
        return
    try:
        msg = MIMEMultipart()
        msg["From"]    = cfg["sender"]
        msg["To"]      = cfg["receiver"]
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
            server.starttls()
            server.login(cfg["sender"], cfg["password"])
            server.send_message(msg)
        print("📧 Email alert sent successfully.")
    except Exception as e:
        print(f"⚠️  Email alert failed: {e}")


def log_violation(image_source: str, violations: list, missing: list):
    cfg = ALERT_CONFIG["log"]
    if not cfg["enabled"]:
        return
    record = {
        "timestamp":   datetime.now().isoformat(),
        "image":       image_source,
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


def trigger_alerts(image_source: str, violations: list, missing: list, status: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"🚨 PPE VIOLATION ALERT",
        f"Time:   {timestamp}",
        f"Image:  {image_source}",
        f"Status: {status}",
    ]
    if violations:
        lines.append("Violations:  " + ", ".join(v.upper() for v in violations))
    if missing:
        lines.append("Missing PPE: " + ", ".join(missing))
    alert_body = "\n".join(lines)

    if ALERT_CONFIG["console"]:
        print("\n" + "!" * 50)
        print(alert_body)
        print("!" * 50)

    send_email_alert(
        subject=f"[PPE Alert] {status} detected at {timestamp}",
        body=alert_body,
    )
    log_violation(image_source, violations, missing)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print(f"\nLoading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Debug: show all class names your model knows
print("\nModel classes:", model.names)

# ─────────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────────
print(f"\nRunning inference on: {IMAGE_PATH}")

results = model.predict(
    source=IMAGE_PATH,
    conf=0.15,
    imgsz=640,
    save=True,
)

print("\n🚨 PPE COMPLIANCE REPORT 🚨")
print("=" * 50)

for r in results:
    detected = []

    print("\nDetected Objects:")
    for box in r.boxes:
        cls_id     = int(box.cls)
        conf       = float(box.conf)
        class_name = model.names[cls_id]
        detected.append(class_name)
        print(f"• {class_name} ({conf:.2f})")

    unique_detected = set(detected)

    print("\nUnique Detected Classes:")
    print(sorted(unique_detected))

    violations_found = [cls for cls in unique_detected if cls in violation_classes]
    missing_ppe      = [ppe for ppe in required_ppe if ppe not in unique_detected]

    print("\n" + "=" * 50)

    if violations_found:
        print("🚨 NON-COMPLIANT WORKER 🚨")
        for v in violations_found:
            print(f"🔴 {v.upper()} DETECTED")
        trigger_alerts(IMAGE_PATH, violations_found, [], "NON-COMPLIANT")

    elif missing_ppe:
        print("⚠️  PPE INCOMPLETE / UNCERTAIN")
        print("\nDetected PPE:")
        for ppe in sorted(required_ppe & unique_detected):
            print(f"✅ {ppe}")
        print("\nMissing PPE:")
        for ppe in sorted(missing_ppe):
            print(f"❌ {ppe}")
        trigger_alerts(IMAGE_PATH, [], missing_ppe, "INCOMPLETE PPE")

    else:
        print("✅ FULLY PPE COMPLIANT — No alert needed.")

print("\nPrediction image saved.")
print("Location: C:/Users/Sneha H S/runs/detect/predict")