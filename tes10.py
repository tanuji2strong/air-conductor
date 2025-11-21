import cv2
import mediapipe as mp
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# --- SETTINGS ---
# Lower this if it's still not detecting (0.02 = 2% of screen height)
MOVEMENT_THRESHOLD = 0.02 
W, H = 640, 480

# --- STATE VARIABLES ---
prev_y = 0
state = "UP" # or "DOWN"
peak_y = 0   # Highest point (smallest Y value)
valley_y = 0 # Lowest point (largest Y value)
last_beat_time = time.time()
bpm = 0
beat_history = []

# For the Graph
y_graph = [H//2] * W

def calculate_bpm(new_beat_time):
    global last_beat_time, beat_history, bpm
    
    delta = new_beat_time - last_beat_time
    last_beat_time = new_beat_time
    
    # Ignore super fast glitches (< 0.2s = 300 BPM)
    if delta > 0.2:
        current_bpm = 60.0 / delta
        beat_history.append(current_bpm)
        if len(beat_history) > 4: 
            beat_history.pop(0)
        bpm = sum(beat_history) / len(beat_history)
        return True
    return False

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, img = cap.read()
        if not ret: break
        
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (W, H))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        flash = False
        
        if results.multi_hand_landmarks:
            # 1. Get Wrist Y
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # y is 0.0 (top) to 1.0 (bottom)
            raw_y = hand_landmarks.landmark[0].y 
            
            # 2. Update Graph Data
            graph_y = int(raw_y * H)
            y_graph.append(graph_y)
            y_graph.pop(0)
            
            # 3. Hysteresis Logic (The "Sticky" Switch)
            # This is much more robust than simple direction checking
            
            if state == "UP":
                # We are moving UP. Look for the Peak (Top).
                if raw_y < peak_y: 
                    peak_y = raw_y # Found a new higher point
                
                # If we drop down significantly below the peak, we switched direction
                if raw_y > peak_y + MOVEMENT_THRESHOLD:
                    state = "DOWN"
                    valley_y = raw_y # Reset valley
                    
            elif state == "DOWN":
                # We are moving DOWN. Look for the Valley (Bottom).
                if raw_y > valley_y:
                    valley_y = raw_y # Found a new lower point
                
                # If we rise up significantly above the valley, THAT IS THE BEAT
                if raw_y < valley_y - MOVEMENT_THRESHOLD:
                    state = "UP"
                    peak_y = raw_y # Reset peak
                    
                    # TRIGGER BEAT
                    if calculate_bpm(time.time()):
                        flash = True

            # Debug Text
            cv2.putText(img, f"State: {state}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # --- VISUALIZATION ---
        
        # Draw the Waveform Graph
        pts = np.array([ [i, val] for i, val in enumerate(y_graph) ], np.int32)
        cv2.polylines(img, [pts], False, (0, 255, 0), 2)
        
        # Draw Threshold Lines (Visual Aid)
        # Allows you to see if you are crossing the threshold
        if results.multi_hand_landmarks:
            curr_y_px = int(raw_y * H)
            thresh_px = int(MOVEMENT_THRESHOLD * H)
            # Draw a little "deadzone" box around the current point
            cv2.rectangle(img, (W-50, curr_y_px - thresh_px), (W, curr_y_px + thresh_px), (50,50,50), 1)

        # Beat Flash
        if flash:
            cv2.circle(img, (50, 50), 40, (0, 255, 255), -1)

        # BPM Display
        cv2.putText(img, f"BPM: {int(bpm)}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        cv2.imshow("Debug Conductor", img)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
