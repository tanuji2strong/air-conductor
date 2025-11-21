import cv2
import mediapipe as mp
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
lineType = cv2.LINE_AA

# ===== BEAT / BPM STATE =====
beat_count = 0
beat_times = []
bpm = 0

# SMOOTHING (Crucial for elderly tremors)
# We average the last 5 Y-positions to remove "jitters"
y_history = deque(maxlen=5)
prev_smoothed_y = 0
direction = 0 # 1 = Down, -1 = Up
last_beat_timestamp = 0

# VISUALS
flash_timer = 0

def compute_bpm(beat_times, window=4):
    """Compute BPM from recent beat timestamps."""
    if len(beat_times) < 2:
        return 0.0
    recent = beat_times[-window:]
    intervals = []
    for i in range(1, len(recent)):
        intervals.append(recent[i] - recent[i - 1])
    if not intervals: return 0.0
    avg_interval = sum(intervals) / len(intervals)
    if avg_interval <= 0: return 0.0
    return 60.0 / avg_interval

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    w, h = 540, 310

    while True:
        ret, img = cap.read()
        if not ret: break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (w, h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        current_time = time.time()

        # Reset BPM if no beat for 2 seconds (User stopped)
        if current_time - last_beat_timestamp > 2.0:
            bpm = 0
            beat_times = []

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # 1. Get Wrist Y (Vertical only)
            wrist_y = hand_landmarks.landmark[0].y
            
            # 2. Smooth the data (Remove Jitter)
            y_history.append(wrist_y)
            smoothed_y = sum(y_history) / len(y_history)

            # 3. Determine Direction (Down or Up?)
            # Y increases downwards in OpenCV
            if smoothed_y > prev_smoothed_y:
                current_dir = 1 # Moving DOWN
            else:
                current_dir = -1 # Moving UP

            # 4. Detect the "Bounce" (The Beat)
            # We were going DOWN (1), now we are going UP (-1)
            if direction == 1 and current_dir == -1:
                # Time since last beat
                delta = current_time - last_beat_timestamp
                
                # Debounce: Ignore beats faster than 200 BPM (0.3s) or slower than 30 BPM (2.0s)
                if 0.3 < delta < 2.0:
                    beat_count += 1
                    beat_times.append(current_time)
                    last_beat_timestamp = current_time
                    bpm = compute_bpm(beat_times)
                    flash_timer = 5 # Trigger visual flash
            
            # Update state
            prev_smoothed_y = smoothed_y
            direction = current_dir

        else:
            cv2.putText(img, "Show Hand", (30, 80), fontFace, 0.8, (0, 0, 255), 2, lineType)

        # --- UI FEEDBACK ---
        
        # Visual Flash (Helps user stay in sync)
        if flash_timer > 0:
            cv2.circle(img, (w-50, 50), 30, (0, 255, 255), -1) # Yellow Flash
            flash_timer -= 1

        cv2.putText(img, f"Beats: {beat_count}", (30, 140), fontFace, 1.0, (255, 255, 255), 2, lineType)
        
        # Color code the BPM (Green = Good speed, Red = Too fast/slow)
        bpm_color = (0, 255, 0)
        if bpm > 140 or bpm < 40 and bpm != 0: bpm_color = (0, 0, 255)
            
        cv2.putText(img, f"BPM: {int(bpm)}", (30, 200), fontFace, 1.5, bpm_color, 3, lineType)

        cv2.imshow("Conductor BPM", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
