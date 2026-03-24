import cv2
import time
import winsound
import os
import sys

# Add parent directory to path for Firebase import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import Firebase manager
try:
    from firebase_config import firebase_manager
    FIREBASE_ENABLED = firebase_manager.initialized
except ImportError:
    print("⚠️ Firebase not configured - Running without cloud storage")
    FIREBASE_ENABLED = False
    firebase_manager = None

# -----------------------------
# Load Haar Cascades
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# Try local file first, then cv2.data
face_cascade_local = os.path.join(script_dir, "haarscascade_frontalface_default.xml")
if os.path.exists(face_cascade_local):
    face_cascade = cv2.CascadeClassifier(face_cascade_local)
else:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

# For eye detection, use cv2.data (standard location)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# -----------------------------
# Parameters
# -----------------------------
EYE_CLOSED_TIME = 2.0   # seconds
closed_start = None
alarm_on = False
total_alerts = 0

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

print("🚀 LocoPilot Drowsiness Detection Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    drowsy = True  # assume eyes closed

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

        if len(eyes) > 0:
            drowsy = False
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                )

    # -----------------------------
    # Drowsiness Logic
    # -----------------------------
    if drowsy:
        if closed_start is None:
            closed_start = time.time()
        elif time.time() - closed_start > EYE_CLOSED_TIME:
            cv2.putText(
                frame,
                "🚨 DROWSINESS ALERT!",
                (80, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                4
            )

            if not alarm_on:
                total_alerts += 1
                winsound.Beep(1200, 1500)
                alarm_on = True
                
                # Save to Firebase
                if FIREBASE_ENABLED and firebase_manager:
                    try:
                        alert_data = {
                            'module': 'drowsiness',
                            'alert_type': 'eyes_closed',
                            'threshold_seconds': EYE_CLOSED_TIME,
                            'alert_number': total_alerts,
                            'status': 'DROWSINESS_ALERT',
                            'mode': 'console_script'
                        }
                        firebase_manager.save_drowsiness_alert(alert_data)
                    except Exception as e:
                        print(f"⚠️ Failed to save to Firebase: {str(e)}")
    else:
        closed_start = None
        alarm_on = False

    cv2.imshow("SmartRail | LocoPilot Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
