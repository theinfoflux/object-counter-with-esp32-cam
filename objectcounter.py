import cv2
import urllib.request
import numpy as np
import time

# ---------------------------
# CONFIG
# ---------------------------
url = 'http://192.168.43.144/'  # Replace with your ESP32-CAM IP
MIN_AREA = 2000                 # Minimum contour area for one candy (adjust if needed)
KERNEL_SIZE = (3, 3)            # Morphology kernel size

# ---------------------------
# Capture background (empty table)
# ---------------------------
print("Capturing background. Make sure the table is empty...")
time.sleep(3)  # Give time to set camera

bg_resp = urllib.request.urlopen(url + 'cam-lo.jpg')
bg_np = np.array(bytearray(bg_resp.read()), dtype=np.uint8)
background = cv2.imdecode(bg_np, cv2.IMREAD_COLOR)
background = cv2.resize(background, (640, 480))
print("Background captured!")

# ---------------------------
# OpenCV windows
# ---------------------------
cv2.namedWindow("Live Candy Counter", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Threshold / Cleaned", cv2.WINDOW_AUTOSIZE)

# ---------------------------
# MAIN LOOP
# ---------------------------
while True:
    # Capture current frame from ESP32-CAM
    img_resp = urllib.request.urlopen(url + 'cam-lo.jpg')
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)
    if frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))

    # ---------------------------
    # Background subtraction
    # ---------------------------
    diff = cv2.absdiff(background, frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_diff, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    # Morphology to clean small details
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ---------------------------
    # Find contours
    # ---------------------------
    cnts, _ = cv2.findContours(clean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for i, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Number each candy
            cv2.putText(frame, str(count), (x + w//2 - 10, y + h//2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # ---------------------------
    # Display total count
    # ---------------------------
    cv2.putText(frame, f"TOTAL OBJECTS: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ---------------------------
    # Show windows
    # ---------------------------
    cv2.imshow("Live Candy Counter", frame)
    cv2.imshow("Threshold / Cleaned", clean)

    # ---------------------------
    # Key handling
    # ---------------------------
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('a'):  # Optional: print count in console
        print("Objects counted:", count)

cv2.destroyAllWindows()
