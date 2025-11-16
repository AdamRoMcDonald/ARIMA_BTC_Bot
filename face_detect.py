import cv2

# ESP32 Camera Stream URL
# (Make sure this matches the one printed in Serial Monitor)
url = "http://192.168.1.38:81/stream"

# Open the ESP32 video stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("ERROR: Could not open ESP32 stream.")
    print("Try pinging the ESP32 from the Pi to confirm connectivity:")
    print("    ping 192.168.1.38")
    exit()

# Load Haar face detector (built into OpenCV)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Face detection running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame from ESP32.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Show the frame
    cv2.imshow("ESP32-CAM Face Detection (Raspberry Pi)", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
