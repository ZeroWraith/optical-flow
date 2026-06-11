import cv2
import numpy as np

# --- STEP 1: OPEN YOUR CAMERA ---
pipeline = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480 ! "
    "videoconvert ! "
    "videoscale ! "
    "video/x-raw, width=640, height=480 ! "
    "appsink"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open camera. Try: sudo apt install gstreamer1.0-libcamera")
    exit()

ret, frame1 = cap.read()
if not ret:
    print("Error: Could not see the camera!")
    exit()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

print("Pi Dense Flow Started! Wave at the camera! Press 'q' to stop.")

while True:
    ret, frame2 = cap.read()
    if not ret: break
    
    next_img = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Farneback Magic (Tracks every pixel live!)
    flow = cv2.calcOpticalFlowFarneback(prvs, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # --- WINDOW 1: RAINBOW MAP ---
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rainbow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- WINDOW 2: ARROW MAP ---
    arrow_img = frame2.copy()
    step = 25 # Slightly bigger steps for speed on the Pi
    for y in range(0, arrow_img.shape[0], step):
        for x in range(0, arrow_img.shape[1], step):
            fx, fy = flow[y, x]
            if abs(fx) > 2 or abs(fy) > 2:
                cv2.arrowedLine(arrow_img, (x, y), (int(x + fx), int(y + fy)), 
                                (0, 255, 0), 1, tipLength=0.5)

    cv2.imshow('Pi Camera - Rainbow Map', rainbow_img)
    cv2.imshow('Pi Camera - Arrow Map', arrow_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prvs = next_img

cap.release()
cv2.destroyAllWindows()
