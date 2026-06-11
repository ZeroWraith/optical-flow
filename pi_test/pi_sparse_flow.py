import cv2
import numpy as np

# --- STEP 1: OPEN YOUR CAMERA ---
# We try to force the camera to use a smaller size immediately 
# to save memory on the Pi.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open camera. Trying GStreamer...")
    pipeline = "libcamerasrc ! video/x-raw, width=640, height=480 ! videoconvert ! appsink"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# Settings for finding good "sticky note" corners
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Settings for the Lucas-Kanade tracking magic
lk_params = dict(winSize=(15, 15), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take the first picture from the camera
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not see the camera!")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

print("Pi Sparse Flow Started! Wave at the camera! Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate where the dots moved
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
        cv2.imshow('Pi Camera - Sparse Flow (Sticky Notes)', img)
        
        # Get ready for the next live frame
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    # If the dots get lost, find new ones!
    if p1 is None or len(good_new) < 10:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
