import cv2
import time
import os
import requests

# =========================
# CONFIG
# =========================

ESP32_IP = "192.168.1.38"          # your ESP32 IP
STREAM_URL = f"http://{ESP32_IP}/stream"
SERVO_URL  = f"http://{ESP32_IP}/servo"

# Center positions
PAN_CENTER  = 90
TILT_CENTER = 90

# YOUR ORIENTATION (do NOT change unless movement is reversed)
PAN_INVERT  = False   # You said increasing pan moves RIGHT
TILT_INVERT = False   # You said increasing tilt moves UP

# Safe limits so you don’t over-rotate the rig
PAN_MIN, PAN_MAX   = 30, 150
TILT_MIN, TILT_MAX = 30, 150

# Speed of movement (degrees per update)
PAN_STEP   = 2
TILT_STEP  = 2

# Delay between servo updates
SERVO_UPDATE_INTERVAL = 0.07

# Save face snapshots (option C)
SAVE_SNAPSHOTS = True
SNAPSHOT_DIR   = os.path.expanduser("~/captures")
SNAPSHOT_INTERVAL = 1.0

# Haar cascade path
CASCADE_PATH = "/usr/share/opencv4/haarcascades_frontalface_default.xml"


# =========================
# SETUP
# =========================

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("ERROR: Failed to load Haar cascade from:", CASCADE_PATH)
    exit(1)

print("Opening ESP32 stream:", STREAM_URL)
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print("ERROR: Could not open ESP32 stream.")
    exit(1)

# Start centered
pan_angle  = PAN_CENTER
tilt_angle = TILT_CENTER

last_servo_update = 0
last_snapshot_time = 0

print("Tracking started. Press 'q' to quit.")


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def send_servo(pan=None, tilt=None):
    params = {}
    if pan is not None: params["pan"] = str(int(pan))
    if tilt is not None: params["tilt"] = str(int(tilt))
    try:
        requests.get(SERVO_URL, params=params, timeout=0.2)
    except:
        pass  # ignore network hiccups


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from ESP32")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2

    # Draw center point
    cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

    target_pan = None
    target_tilt = None

    if len(faces) > 0:
        # Largest face = closest
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, fw, fh) = faces[0]
        fx = x + fw // 2
        fy = y + fh // 2

        # Draw face box + center
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        cv2.circle(frame, (fx, fy), 4, (0, 255, 0), -1)

        # How far off center
        error_x = fx - center_x
        error_y = fy - center_y

        DEADZONE_X = w * 0.07
        DEADZONE_Y = h * 0.07

        now = time.time()

        # Save snapshot
        if SAVE_SNAPSHOTS and now - last_snapshot_time > SNAPSHOT_INTERVAL:
            filename = os.path.join(SNAPSHOT_DIR, f"face_{int(now)}.jpg")
            cv2.imwrite(filename, frame)
            print("Saved snapshot:", filename)
            last_snapshot_time = now

        # Servo update timing
        if now - last_servo_update > SERVO_UPDATE_INTERVAL:

            # -------- PAN CONTROL --------
            if abs(error_x) > DEADZONE_X:
                direction = 1 if error_x > 0 else -1  # face right => increase pan
                if PAN_INVERT: direction *= -1
                pan_angle += direction * PAN_STEP
                pan_angle = clamp(pan_angle, PAN_MIN, PAN_MAX)
                target_pan = pan_angle

            # -------- TILT CONTROL --------
            if abs(error_y) > DEADZONE_Y:
                direction = 1 if error_y > 0 else -1  # face below => increase tilt
                if TILT_INVERT: direction *= -1
                tilt_angle += direction * TILT_STEP
                tilt_angle = clamp(tilt_angle, TILT_MIN, TILT_MAX)
                target_tilt = tilt_angle

            send_servo(target_pan, target_tilt)
            last_servo_update = now

    cv2.imshow("ESP32 Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
