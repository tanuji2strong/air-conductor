import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION ---
THRESHOLD = 0.04  # vertical pull-up distance to count 1 beat (normalized 0~1)

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# --- STATE ---
state = "WAITING"  # WAITING -> DOWN -> UP
lowest_y = 0.0

beat_count = 0
last_beat_time = None
bpm = 0
flash_timer = 0

# 4/4 bar position: 1,2,3,4
beat_in_bar = 1

PATTERN_TEXT = {
    1: "Beat 1: DOWN (strong)",
    2: "Beat 2: LEFT",
    3: "Beat 3: RIGHT",
    4: "Beat 4: UP (prep)"
}

def draw_44_pattern(img, cx, cy, beat_in_bar):
    """
    Draw 4/4 conductor pattern with pivot at (cx, cy).
    Highlight current beat direction.
    """
    L = 150  # length of strokes in pixels
    base_color = (80, 80, 80)
    hi_color   = (0, 255, 255)
    thick_base = 2
    thick_hi   = 6

    # DOWN stroke (Beat 1)
    down_start = (cx, cy)
    down_end   = (cx, cy + L)
    # LEFT stroke (Beat 2)
    left_start = down_end
    left_end   = (cx - L//2, cy)
    # RIGHT stroke (Beat 3)
    right_start = left_end
    right_end   = (cx + L//2, cy)
    # UP stroke (Beat 4)
    up_start = right_end
    up_end   = (cx, cy)

    # draw base pattern first
    cv2.line(img, down_start, down_end, base_color, thick_base)
    cv2.line(img, left_start, left_end, base_color, thick_base)
    cv2.line(img, right_start, right_end, base_color, thick_base)
    cv2.line(img, up_start, up_end, base_color, thick_base)

    # highlight current beat segment
    if beat_in_bar == 1:
        cv2.line(img, down_start, down_end, hi_color, thick_hi)
    elif beat_in_bar == 2:
        cv2.line(img, left_start, left_end, hi_color, thick_hi)
    elif beat_in_bar == 3:
        cv2.line(img, right_start, right_end, hi_color, thick_hi)
    elif beat_in_bar == 4:
        cv2.line(img, up_start, up_end, hi_color, thick_hi)

    # optional: number markers at ends
    cv2.putText(img, "1", down_end, font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "2", left_end, font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "3", right_end, font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "4", up_end, font, 0.8, (255, 255, 255), 2)


with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=1) as hands:

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        wrist_px, wrist_py = None, None
        curr_y = None
        flash = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            # wrist normalized coords
            wrist = hand.landmark[0]
            curr_y = wrist.y  # normalized 0(top)~1(bottom)

            wrist_px = int(wrist.x * w)
            wrist_py = int(wrist.y * h)

            # --- STICKY DOWN/UP BEAT LOGIC ---

            if state == "WAITING":
                state = "DOWN"
                lowest_y = curr_y

            elif state == "DOWN":
                # update lowest (most downward) Y
                if curr_y > lowest_y:
                    lowest_y = curr_y

                # pulled up enough from lowest point -> beat
                if curr_y < (lowest_y - THRESHOLD):
                    state = "UP"
                    beat_count += 1
                    flash = True

                    now = time.time()
                    if last_beat_time is not None:
                        delta = now - last_beat_time
                        if 0.3 < delta < 2.5:  # sane human range
                            bpm = 60.0 / delta
                    last_beat_time = now

                    # advance 4/4 bar: 1->2->3->4->1
                    beat_in_bar += 1
                    if beat_in_bar > 4:
                        beat_in_bar = 1

            elif state == "UP":
                # After rebound, once hand is clearly higher than lowest zone, re-arm DOWN
                if curr_y < (lowest_y - THRESHOLD / 2):
                    # high enough, ready for next downstroke
                    state = "DOWN"
                    lowest_y = curr_y

            # Draw threshold line when in DOWN state
            if state == "DOWN":
                trigger_y = int((lowest_y - THRESHOLD) * h)
                if 0 <= trigger_y < h:
                    cv2.line(img, (0, trigger_y), (w, trigger_y), (0, 255, 255), 2)
                    cv2.putText(img, "Pull UP past line", (10, trigger_y - 10),
                                font, 0.7, (0, 255, 255), 2)

        else:
            state = "WAITING"
            lowest_y = 0.0

        # --- VISUALS ---

        # 1. Flash on beat
        if flash_timer > 0:
            flash_timer -= 1
        if flash:
            flash_timer = 5

        if flash_timer > 0:
            cv2.circle(img, (50, 50), 40, (0, 255, 0), -1)

        # 2. Draw 4/4 pattern around wrist pivot (if hand found)
        if wrist_px is not None and wrist_py is not None:
            draw_44_pattern(img, wrist_px, wrist_py, beat_in_bar)

        # 3. Text HUD
        status_color = (0, 255, 0) if state == "UP" else (0, 0, 255)
        cv2.putText(img, f"State: {state}", (30, 80), font, 1, status_color, 2)
        cv2.putText(img, f"BPM: {int(bpm)}", (30, 140), font, 1.5, (255, 255, 255), 3)
        cv2.putText(img, f"Total Beats: {beat_count}", (30, 190), font, 1, (200, 200, 200), 2)
        cv2.putText(img, PATTERN_TEXT[beat_in_bar], (30, 240), font, 0.9, (255, 255, 0), 2)
        cv2.putText(img, "Bounce down for each beat, follow the pattern with your wrist.",
                    (30, h - 40), font, 0.7, (200, 200, 200), 2)

        cv2.imshow("Sticky 4/4 Conductor", img)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
