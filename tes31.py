import cv2
import mediapipe as mp
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION ---
THRESHOLD = 0.05      # Sensitivity for the vertical "pull up"
COUNTDOWN_SECONDS = 3
SMOOTHING_WINDOW = 5  # Number of beats to average for BPM

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# --- STATE ---
state = "WAITING"  # WAITING -> DOWN -> UP
lowest_y = 0.0

beat_count = 0
last_beat_time = None
bpm_history = deque(maxlen=SMOOTHING_WINDOW)
current_bpm = 0

bar_count = 0
beat_in_bar = 1  # 1, 2, 3, 4

flash_timer = 0

PATTERN_TEXT = {
    1: "1: DOWN (Strong)",
    2: "2: LEFT (In)",
    3: "3: RIGHT (Out)",
    4: "4: UP (Prep)"
}

def draw_static_overlay(img, cx, cy, active_beat):
    """
    Draws the 4/4 pattern in the CENTER of the screen (or fixed point).
    The user moves their hand relative to this, rather than it moving with them.
    """
     # Conceptual tag
    
    L = 120 # Length of lines
    base_color = (60, 60, 60)
    active_color = (0, 255, 255)
    
    # Coordinates relative to center (cx, cy)
    center = (cx, cy)
    down_pt = (cx, cy + L)
    left_pt = (cx - L, cy)
    right_pt = (cx + L, cy)
    up_pt = (cx, cy - L//2) # Visual marker for up

    # Draw Base Cross
    cv2.line(img, (cx, cy - L), (cx, cy + L), base_color, 2)
    cv2.line(img, (cx - L, cy), (cx + L, cy), base_color, 2)

    # Highlight the "Target" zone for the current beat
    if active_beat == 1: # Target is DOWN
        cv2.line(img, center, down_pt, active_color, 6)
        cv2.circle(img, down_pt, 10, active_color, -1)
    elif active_beat == 2: # Target is LEFT
        cv2.line(img, center, left_pt, active_color, 6)
        cv2.circle(img, left_pt, 10, active_color, -1)
    elif active_beat == 3: # Target is RIGHT
        cv2.line(img, center, right_pt, active_color, 6)
        cv2.circle(img, right_pt, 10, active_color, -1)
    elif active_beat == 4: # Target is UP
        cv2.line(img, center, up_pt, active_color, 6)
        cv2.circle(img, up_pt, 10, active_color, -1)

    # Labels
    cv2.putText(img, "1", (cx + 10, cy + L), font, 0.6, (150,150,150), 1)
    cv2.putText(img, "2", (cx - L, cy - 10), font, 0.6, (150,150,150), 1)
    cv2.putText(img, "3", (cx + L - 20, cy - 10), font, 0.6, (150,150,150), 1)
    cv2.putText(img, "4", (cx + 10, cy - L//2), font, 0.6, (150,150,150), 1)


def run_countdown():
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        elapsed = time.time() - start
        remain = COUNTDOWN_SECONDS - int(elapsed)
        
        if remain > 0:
            text = str(remain)
            color = (0, 255, 255)
        else:
            text = "CONDUCT!"
            color = (0, 255, 0)

        cv2.putText(frame, text, (w//2 - 100, h//2), font, 3, color, 5)
        cv2.imshow("Conductor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): return False
        if elapsed >= COUNTDOWN_SECONDS + 1: break
    return True

# --- MAIN LOOP ---
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    
    if run_countdown():
        
        while cap.isOpened():
            ret, img = cap.read()
            if not ret: break
            
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            # Draw static overlay in the center of the screen
            draw_static_overlay(img, w//2, h//2, beat_in_bar)

            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

                # Index finger tip
                idx = hand.landmark[8]
                curr_y = idx.y
                
                # --- LOGIC: Vertical Schmitt Trigger ---
                if state == "WAITING":
                    state = "DOWN"
                    lowest_y = curr_y

                elif state == "DOWN":
                    # Track the lowest point reached in this stroke
                    if curr_y > lowest_y:
                        lowest_y = curr_y
                    
                    # Check for upward pull (Rebound)
                    if curr_y < (lowest_y - THRESHOLD):
                        state = "UP"
                        flash_timer = 4
                        
                        # Calculate BPM
                        now = time.time()
                        if last_beat_time:
                            delta = now - last_beat_time
                            if 0.2 < delta < 2.0: # Filter unreasonable speeds
                                inst_bpm = 60.0 / delta
                                bpm_history.append(inst_bpm)
                                current_bpm = sum(bpm_history) / len(bpm_history)
                        
                        last_beat_time = now
                        beat_count += 1
                        
                        # Cycle the bar
                        beat_in_bar += 1
                        if beat_in_bar > 4:
                            beat_in_bar = 1
                            bar_count += 1

                elif state == "UP":
                    # Reset logic: once hand goes up enough, re-arm for next downbeat
                    # Or if hand starts moving down significantly
                    if curr_y > (lowest_y - (THRESHOLD * 0.5)): 
                        state = "DOWN"
                        lowest_y = curr_y # Reset floor

                # Visual Helper for Threshold
                if state == "DOWN":
                    trigger_y = int((lowest_y - THRESHOLD) * h)
                    cv2.line(img, (0, trigger_y), (w, trigger_y), (0, 100, 255), 1)

            else:
                # Reset if hand lost
                state = "WAITING"

            # --- HUD ---
            if flash_timer > 0:
                cv2.circle(img, (w - 50, 50), 30, (0, 255, 0), -1)
                flash_timer -= 1
            
            cv2.putText(img, f"BPM: {int(current_bpm)}", (20, 50), font, 1.2, (255, 255, 255), 2)
            cv2.putText(img, f"Beat: {beat_in_bar} / 4", (20, 100), font, 1, (0, 255, 255), 2)
            cv2.putText(img, PATTERN_TEXT[beat_in_bar], (20, h - 30), font, 0.8, (200, 200, 200), 2)

            cv2.imshow("Conductor", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
