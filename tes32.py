import cv2
import mediapipe as mp
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION ---
THRESHOLD = 0.05        # Vertical "pull up" (normalized 0~1)
COUNTDOWN_SECONDS = 3
SMOOTHING_WINDOW = 5    # Number of recent beats for BPM averaging

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# --- STATE ---
state = "WAITING"       # WAITING -> DOWN -> UP
lowest_y = 0.0

beat_count = 0
last_beat_time = None
bpm_history = deque(maxlen=SMOOTHING_WINDOW)
current_bpm = 0.0

bar_count = 0
beat_in_bar = 1         # 1, 2, 3, 4

flash_timer = 0

# For Y smoothing
y_history = deque(maxlen=5)
prev_smooth_y = None
VERT_EPS = 0.003        # ignore tiny vertical changes

PATTERN_TEXT = {
    1: "1: DOWN (Strong)",
    2: "2: LEFT (In)",
    3: "3: RIGHT (Out)",
    4: "4: UP (Prep)"
}

def draw_static_overlay(img, cx, cy, active_beat):
    """
    Draw 4/4 pattern fixed in the center of the screen.
    User moves relative to this.
    """
    L = 120  # line length
    base_color = (60, 60, 60)
    active_color = (0, 255, 255)

    center = (cx, cy)
    down_pt  = (cx,     cy + L)
    left_pt  = (cx - L, cy)
    right_pt = (cx + L, cy)
    up_pt    = (cx,     cy - L//2)

    # Base cross
    cv2.line(img, (cx, cy - L), (cx, cy + L), base_color, 2)
    cv2.line(img, (cx - L, cy), (cx + L, cy), base_color, 2)

    # Highlight current beat
    if active_beat == 1:
        cv2.line(img, center, down_pt, active_color, 6)
        cv2.circle(img, down_pt, 10, active_color, -1)
    elif active_beat == 2:
        cv2.line(img, center, left_pt, active_color, 6)
        cv2.circle(img, left_pt, 10, active_color, -1)
    elif active_beat == 3:
        cv2.line(img, center, right_pt, active_color, 6)
        cv2.circle(img, right_pt, 10, active_color, -1)
    elif active_beat == 4:
        cv2.line(img, center, up_pt, active_color, 6)
        cv2.circle(img, up_pt, 10, active_color, -1)

    # Labels
    cv2.putText(img, "1", (cx + 10, cy + L), font, 0.6, (150, 150, 150), 1)
    cv2.putText(img, "2", (cx - L, cy - 10), font, 0.6, (150, 150, 150), 1)
    cv2.putText(img, "3", (cx + L - 20, cy - 10), font, 0.6, (150, 150, 150), 1)
    cv2.putText(img, "4", (cx + 10, cy - L//2), font, 0.6, (150, 150, 150), 1)


def run_countdown():
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        text_size, _ = cv2.getTextSize(text, font, 3, 5)
        tw, th = text_size
        cv2.putText(frame, text, (w//2 - tw//2, h//2 + th//2),
                    font, 3, color, 5)
        cv2.putText(frame, "Raise your index and get ready...",
                    (20, h - 40), font, 0.8, (200, 200, 200), 2)

        cv2.imshow("Conductor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        if elapsed >= COUNTDOWN_SECONDS + 1:
            break
    return True


with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=1
) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    if run_countdown():

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.flip(img, 1)
            h, w, _ = img.shape

            # Static 4/4 overlay (centered)
            draw_static_overlay(img, w // 2, h // 2, beat_in_bar)

            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

                # Index finger tip
                idx = hand.landmark[8]
                raw_y = idx.y  # normalized [0,1]

                # ---- SMOOTHING ----
                y_history.append(raw_y)
                smooth_y = sum(y_history) / len(y_history)

                # Initialize smoothing baseline
                if prev_smooth_y is None:
                    prev_smooth_y = smooth_y

                # ---- STATE MACHINE (using smoothed Y) ----
                if state == "WAITING":
                    state = "DOWN"
                    lowest_y = smooth_y

                elif state == "DOWN":
                    # Update lowest point
                    if smooth_y > lowest_y:
                        lowest_y = smooth_y

                    # Only consider a real upward pull if we've moved more than epsilon
                    dy = smooth_y - prev_smooth_y
                    # upwards movement is dy < 0
                    if smooth_y < (lowest_y - THRESHOLD) and dy < -VERT_EPS:
                        state = "UP"
                        flash_timer = 4

                        now = time.time()
                        if last_beat_time:
                            delta = now - last_beat_time
                            if 0.2 < delta < 2.0:
                                inst_bpm = 60.0 / delta
                                bpm_history.append(inst_bpm)
                                current_bpm = sum(bpm_history) / len(bpm_history)
                        last_beat_time = now

                        beat_count += 1

                        # Advance bar position
                        beat_in_bar += 1
                        if beat_in_bar > 4:
                            beat_in_bar = 1
                            bar_count += 1

                elif state == "UP":
                    # Re-arm DOWN once we've gone high enough (or stopped rising)
                    # Essentially: when we've clearly moved away from the bottom region
                    if smooth_y < (lowest_y - THRESHOLD * 0.5):
                        state = "DOWN"
                        lowest_y = smooth_y

                # Visual helper: threshold line
                if state == "DOWN":
                    trigger_y = int((lowest_y - THRESHOLD) * h)
                    cv2.line(img, (0, trigger_y), (w, trigger_y), (0, 100, 255), 1)

                prev_smooth_y = smooth_y

            else:
                # Lost hand -> reset state gradually
                state = "WAITING"
                lowest_y = 0.0
                prev_smooth_y = None
                y_history.clear()

            # --- HUD / Feedback ---
            if flash_timer > 0:
                cv2.circle(img, (w - 50, 50), 30, (0, 255, 0), -1)
                flash_timer -= 1

            cv2.putText(img, f"BPM: {int(current_bpm)}", (20, 50), font, 1.2, (255, 255, 255), 2)
            cv2.putText(img, f"Beat: {beat_in_bar} / 4", (20, 100), font, 1, (0, 255, 255), 2)
            cv2.putText(img, f"Total Beats: {beat_count} | Bars: {bar_count}",
                        (20, 140), font, 0.8, (200, 200, 200), 2)
            cv2.putText(img, PATTERN_TEXT[beat_in_bar],
                        (20, h - 30), font, 0.8, (200, 200, 200), 2)

            cv2.imshow("Conductor", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
