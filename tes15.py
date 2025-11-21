import cv2
import mediapipe as mp
import time
import math
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

WINDOW_NAME = "Forgiving Conductor"
cap = cv2.VideoCapture(0)
fontFace = cv2.FONT_HERSHEY_SIMPLEX

# ===== STATE =====
beat_count = 0
beat_times = []
beat_in_bar = 1 # 1, 2, 3, 4

# SMOOTHING (Crucial for elderly)
y_history = deque(maxlen=5)
prev_smoothed_y = 0
direction = 0 # 1 = Down, -1 = Up
last_beat_time = 0
bpm = 0

# 4/4 Pattern (For visual guidance only)
PATTERN_TEXT = {
    1: "DOWN (Strong!)",
    2: "Left (In)",
    3: "Right (Out)",
    4: "Up (Prep)"
}

def draw_arrow(img, beat_num):
    """Visual suggestion of where to move, but not enforced."""
    h, w, _ = img.shape
    cx, cy = w // 2, h // 2
    length = 100
    color = (0, 255, 255) # Yellow
    
    if beat_num == 1: # Down
        cv2.arrowedLine(img, (cx, cy-length), (cx, cy+length), color, 5, 0, 0, 0.2)
    elif beat_num == 2: # Left
        cv2.arrowedLine(img, (cx+length, cy), (cx-length, cy), color, 5, 0, 0, 0.2)
    elif beat_num == 3: # Right
        cv2.arrowedLine(img, (cx-length, cy), (cx+length, cy), color, 5, 0, 0, 0.2)
    elif beat_num == 4: # Up
        cv2.arrowedLine(img, (cx, cy+length), (cx, cy-length), color, 5, 0, 0, 0.2)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, img = cap.read()
        if not ret: break
        
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (960, 540))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        flash = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            
            # 1. TRACK VERTICAL BOUNCE (The Engine)
            wrist_y = hand.landmark[0].y
            y_history.append(wrist_y)
            smoothed_y = sum(y_history) / len(y_history)
            
            # Determine Vertical Direction
            if smoothed_y > prev_smoothed_y:
                curr_dir = 1 # Down
            else:
                curr_dir = -1 # Up
            
            # 2. DETECT BEAT (Bottom of the bounce)
            # If we were going DOWN, and now UP -> That's a beat.
            if direction == 1 and curr_dir == -1:
                curr_time = time.time()
                delta = curr_time - last_beat_time
                
                # Debounce (limit speed to human range)
                if 0.3 < delta < 2.5:
                    beat_count += 1
                    beat_times.append(curr_time)
                    last_beat_time = curr_time
                    
                    # Calculate BPM
                    if len(beat_times) > 1:
                        bpm = 60.0 / delta
                    
                    # AUTO-ADVANCE THE BAR
                    # We don't care if they moved Left or Right. 
                    # A bounce is a bounce. We advance the music.
                    beat_in_bar += 1
                    if beat_in_bar > 4: beat_in_bar = 1
                    
                    flash = True # Visual Flash
            
            prev_smoothed_y = smoothed_y
            direction = curr_dir

        # --- DRAW UI ---
        # 1. The "Guidance" Arrow (For the NEXT beat)
        draw_arrow(img, beat_in_bar)
        
        # 2. Visual Flash on Beat
        if flash:
            cv2.circle(img, (50, 50), 40, (0, 255, 0), -1)
        
        # 3. Text Info
        cv2.putText(img, f"BPM: {int(bpm)}", (30, 100), fontFace, 1.5, (255, 255, 255), 3)
        cv2.putText(img, f"Pattern: {PATTERN_TEXT[beat_in_bar]}", (30, 150), fontFace, 1, (255, 255, 0), 2)
        cv2.putText(img, "Just bounce your hand!", (30, 500), fontFace, 0.8, (200, 200, 200), 2)

        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(5) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
