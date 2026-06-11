import cv2
import numpy as np
from picamera2 import Picamera2

# --- STEP 1: OPEN YOUR CAMERA WITH PICAMERA2 ---
print("Starting Picamera2...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

# Prepare for Dense Flow
frame1 = picam2.capture_array()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

print("Pi Dense Flow Started! Press 'q' to stop.")

try:
    while True:
        frame2 = picam2.capture_array()
        next_img = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Farneback Magic
        flow = cv2.calcOpticalFlowFarneback(prvs, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Rainbow Map
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rainbow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Arrow Map
        arrow_img = frame2.copy()
        step = 25
        for y in range(0, arrow_img.shape[0], step):
            for x in range(0, arrow_img.shape[1], step):
                fx, fy = flow[y, x]
                if abs(fx) > 2 or abs(fy) > 2:
                    cv2.arrowedLine(arrow_img, (x, y), (int(x + fx), int(y + fy)), 
                                    (0, 255, 0), 1, tipLength=0.5)

        cv2.imshow('AI Camera - Rainbow Map', rainbow_img)
        cv2.imshow('AI Camera - Arrow Map', arrow_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prvs = next_img
finally:
    picam2.stop()
    cv2.destroyAllWindows()
