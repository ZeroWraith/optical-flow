import cv2
import numpy as np

# --- STEP 1: LOAD THE MOVIE ---
cap = cv2.VideoCapture('lucas-kanade-optical-flow-demo.mp4')

# --- STEP 2: PREPARE THE FIRST PICTURE ---
ret, frame1 = cap.read()
if not ret:
    print("Error: Could not read video.")
    exit()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Prepare the "Rainbow Map" settings (HSV)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255 

print("Starting the Double Demo!")
print("Window 1: Rainbow Tracker (Color shows direction)")
print("Window 2: Arrow Tracker (Little pointers show direction)")
print("Press 'q' to stop.")

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    
    next_img = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # --- STEP 3: TRACK EVERY PIXEL (The Farneback Magic) ---
    flow = cv2.calcOpticalFlowFarneback(prvs, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # --- WINDOW 1: MAKE THE RAINBOW MAP ---
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rainbow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- WINDOW 2: MAKE THE ARROW MAP ---
    # We start with the original video frame
    arrow_img = frame2.copy()
    
    # We don't want too many arrows (it would be a mess!), 
    # so we only draw one every 20 pixels.
    step = 20
    for y in range(0, arrow_img.shape[0], step):
        for x in range(0, arrow_img.shape[1], step):
            # Get the movement (flow) at this spot
            fx, fy = flow[y, x]
            
            # Only draw an arrow if it's actually moving a bit
            if abs(fx) > 1 or abs(fy) > 1:
                # Draw a little arrow from the old spot to the new spot
                cv2.arrowedLine(arrow_img, (x, y), (int(x + fx), int(y + fy)), 
                                (0, 255, 0), 1, tipLength=0.5)

    # Show both windows!
    cv2.imshow('1. Rainbow Tracker (Direction = Color)', rainbow_img)
    cv2.imshow('2. Arrow Tracker (Direction = Arrows)', arrow_img)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    prvs = next_img

cap.release()
cv2.destroyAllWindows()
