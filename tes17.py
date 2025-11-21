import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION ---
# How much hand needs to move up to trigger a beat (0.05 = 5% of screen height)
# If it's not detecting, LOWER this to 0.03
# If it detects too many false beats, RAISE this to 0.07
THRESHOLD = 0.04 

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Variables
state = "WAITING"  # WAITING -> DOWN -> UP
lowest_y = 0       # To track the bottom of the beat
beat_count = 0
last_beat_time = time.time()
bpm = 0
flash_timer = 0

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, img = cap.read()
        if not ret: break
        
        # 1. Setup Image
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        img = cv2.resize(img, (w, h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        curr_y = 0
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            
            # Use Wrist Y-position (Normalized 0.0 top to 1.0 bottom)
            curr_y = hand.landmark[0].y 
            
            # --- THE STICKY LOGIC ---
            
            # Case 1: We are moving DOWN (preparing the beat)
            if state == "DOWN":
                # Track the lowest point we reach
                if curr_y > lowest_y:
                    lowest_y = curr_y
                
                # CHECK: Have we pulled UP enough to count as a beat?
                # We need to rise higher than (lowest_point - threshold)
                if curr_y < (lowest_y - THRESHOLD):
                    # BEAT DETECTED!
                    state = "UP"
                    beat_count += 1
                    flash_timer = 5
                    
                    # Calculate BPM
                    now = time.time()
                    delta = now - last_beat_time
                    if 0.3 < delta < 2.5: # Filter crazy speeds
                        bpm = 60.0 / delta
                    last_beat_time = now

            # Case 2: We are moving UP (rebound)
            elif state == "UP":
                # Reset logic: Once hand goes high enough, we are ready to go down again
                # (Simple reset when hand is in upper half of movement range)
                # Or just switch to DOWN if we start dropping?
                
                # Let's simply switch to DOWN if we drop slightly from a local peak
                # For simplicity in this demo, let's just assume if we moved enough UP, 
                # we are ready to go DOWN.
                
                # Actually, let's just look for a drop.
                if curr_y > (lowest_y - THRESHOLD/2): 
                    # Use a smaller threshold to detect start of drop
                    pass 
                else:
                    # We are high enough, reset state to catch next downstroke
                    state = "DOWN"
                    lowest_y = 0 # Reset baseline
            
            # Initial Start
            elif state == "WAITING":
                state = "DOWN"
                lowest_y = curr_y

            # --- VISUAL DEBUG BARS (Crucial for you to see) ---
            # Draw the Threshold Line
            # This line shows WHERE you need to pull your hand up to
            if state == "DOWN":
                trigger_y = int((lowest_y - THRESHOLD) * h)
                cv2.line(img, (0, trigger_y), (w, trigger_y), (0, 255, 255), 2)
                cv2.putText(img, "PULL UP PAST LINE", (10, trigger_y - 10), font, 0.7, (0, 255, 255), 2)
            
        # --- UI FEEDBACK ---
        if flash_timer > 0:
            cv2.circle(img, (50, 50), 40, (0, 255, 0), -1)
            flash_timer -= 1
            
        status_color = (0, 255, 0) if state == "UP" else (0, 0, 255)
        cv2.putText(img, f"State: {state}", (30, 80), font, 1, status_color, 2)
        cv2.putText(img, f"BPM: {int(bpm)}", (30, 140), font, 1.5, (255, 255, 255), 3)
        cv2.putText(img, f"Total Beats: {beat_count}", (30, 190), font, 1, (200, 200, 200), 2)

        cv2.imshow("Sticky Conductor", img)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
