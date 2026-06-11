import cv2
import numpy as np
from picamera2 import Picamera2

# --- STEP 1: OPEN YOUR CAMERA WITH PICAMERA2 ---
# This is the modern way for Pi 5 and the AI Camera!
print("Starting Picamera2...")
picam2 = Picamera2()

# Configure the camera for a small, fast stream
config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

# Settings for finding good "sticky note" corners
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Settings for the Lucas-Kanade tracking magic
lk_params = dict(winSize=(15, 15), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take the first picture
old_frame = picam2.capture_array()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

print("Pi Sparse Flow Started! Press 'q' to stop.")

try:
    while True:
        # Capture a frame directly into a numpy array (OpenCV format)
        frame = picam2.capture_array()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate where the dots moved
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw the trails
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

                img = cv2.add(frame, mask)
                cv2.imshow('AI Camera - Sparse Flow', img)
                
                # Get ready for the next live frame
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
        
        # If we lose points, find new ones
        if p0 is None or len(p0) < 10:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
