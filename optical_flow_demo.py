import cv2
import numpy as np

# --- STEP 1: LOAD THE MOVIE ---
# We open the video file you have in your folder.
cap = cv2.VideoCapture('lucas-kanade-optical-flow-demo.mp4')

# --- STEP 2: PICK THE DOTS WE WANT TO FOLLOW ---
# Imagine we are putting sticky notes on "corners" or sharp edges 
# because they are the easiest things for the computer to see.
feature_params = dict(maxCorners=10,      # We only want to track up to 100 sticky notes
                       qualityLevel=0.3,    # Only pick "good" clear corners
                       minDistance=7,       # Don't put sticky notes too close together
                       blockSize=7)         # Look at a small square around the point

# --- STEP 3: THE "MAGIC" TRACKING SETTINGS ---
# This is how the "Lucas-Kanade" algorithm works. 
# It's like a magnifying glass that looks for where our sticky notes went!
lk_params = dict(winSize=(15, 15),     # How big of a search area to look in
                  maxLevel=2,          # Look at the picture in "blurry" mode first to see big moves
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# --- STEP 4: PREPARE THE FIRST PICTURE ---
# Read the very first frame of the video.
ret, old_frame = cap.read()

if not ret:
    print("Error: Could not read the video file. Make sure 'lucas-kanade-optical-flow-demo.mp4' is in the folder.")
    exit()

# Turn it grey (like an old TV) because colors make the math too hard for now!
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# Find those "sticky note" points in the first frame.
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a "blank drawing paper" the same size as the video. 
# We will draw the colorful trails on this.
mask = np.zeros_like(old_frame)

print("Starting the demo! Press 'q' on your keyboard to stop it.")

while True:
    # Read the next frame of the video.
    ret, frame = cap.read()
    if not ret:
        break # Stop if the movie ends
    
    # Turn the new frame grey too.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- STEP 5: THE TRACKING HAPPENS HERE! ---
    # This function asks: "Where did our points (p0) move to in the new picture (frame_gray)?"
    # p1 = the new spots, st = 'status' (did we find it?), err = how sure we are.
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Only keep the points that the computer actually found.
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # --- STEP 6: DRAW THE TRAILS ---
        # We loop through each point to draw a line from where it WAS to where it IS.
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel() # New spot (x, y)
            c, d = old.ravel() # Old spot (x, y)
            
            # Draw a colorful line on our "blank paper" (mask).
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            # Draw a little dot on the current frame.
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # Mix the video frame and our "drawing paper" together.
        img = cv2.add(frame, mask)

        # Show the result on the screen!
        cv2.imshow('Lucas-Kanade Optical Flow (The Magic Tracker)', img)
        
        # --- STEP 7: GET READY FOR THE NEXT FRAME ---
        # Now, the current "new" frame becomes the "old" frame for the next step.
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    # Wait for a tiny bit (30ms). If you press 'q', we quit.
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Clean up and close all windows.
cap.release()
cv2.destroyAllWindows()
