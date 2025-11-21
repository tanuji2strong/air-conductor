import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION ---
THRESHOLD = 0.04      # vertical pull-up distance to count 1 beat (normalized 0~1)
COUNTDOWN_SECONDS = 3

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# --- STATE ---
state = "WAITING"  # WAITING -> DOWN -> UP
lowest_y = 0.0

beat_count = 0          # single beats (down-up bounces)
last_beat_time = None
bpm = 0                 # beats per minute (quarter-note level)

bar_count = 0           # full 4/4 cycles (Down-Left-Right-Up)
last_bar_time = None
bar_bpm = 0             # bars per minute (one full pattern)

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
    """Draw 4/4 conductor pattern pivoted at (cx, cy)."""
    L = 150
    base = (80, 80, 80)
    hi   = (0, 255, 255)
    tb, th = 2, 6

    # DOWN
    d1 = (cx, cy)
    d2 = (cx, cy + L)
    # LEFT
    l1 = d2
    l2 = (cx - L//2, cy)
    # RIGHT
    r1 = l2
    r2 = (cx + L//2, cy)
    # UP
    u1 = r2
    u2 = (cx, cy)

    # base lines
    cv2.line(img, d1, d2, base, tb)
    cv2.line(img, l1, l2, base, tb)
    cv2.line(img, r1, r2, base, tb)
    cv2.line(img, u1, u2, base, tb)

    # highlight current stroke
    if beat_in_bar == 1:
        cv2.line(img, d1, d2, hi, th)
    elif beat_in_bar == 2:
        cv2.line(img, l1, l2, hi, th)
    elif beat_in_bar == 3:
        cv2.line(img, r1, r2, hi, th)
    elif beat_in_bar == 4:
        cv2.line(img, u1, u2, hi, th)

    # markers
    cv2.putText(img, "1", d2, font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "2", l2, font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "3", r2, font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "4", u2, font, 0.8, (255, 255, 255), 2)


def run_countdown():
    """Show 3-2-1-GO countdown before starting detection."""
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
        else:
            text = "GO"

        size, _ = cv2.getTextSize(text, font, 3, 6)
        tw, th = size
        cx, cy = w // 2, h // 2

        cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                    font, 3, (0, 255, 0), 6)

        cv2.putText(frame, "Get your hand (index finger) in position...",
                    (30, h - 40), font, 0.8, (200, 200, 200), 2)

        cv2.imshow("Sticky 4/4 Conductor", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            return False

        if elapsed >= COUNTDOWN_SECONDS + 0.8:
            break

    return True


with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # ---- COUNTDOWN ----
    if not run_countdown():
        cap.release()
        cv2.destroyAllWindows()
        exit()

    global state, lowest_y, beat_count, last_beat_time, bpm
    global bar_count, last_bar_time, bar_bpm, flash_timer, beat_in_bar

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        index_px, index_py = None, None
        curr_y = None
        flash = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            # INDEX FINGER TIP as pivot (landmark 8)
            idx = hand.landmark[8]
            curr_y = idx.y
            index_px = int(idx.x * w)
            index_py = int(idx.y * h)

            # ---- Sticky vertical beat detection ----
            if state == "WAITING":
                state = "DOWN"
                lowest_y = curr_y

            elif state == "DOWN":
                if curr_y > lowest_y:
                    lowest_y = curr_y

                # pulled up enough from bottom -> beat
                if curr_y < (lowest_y - THRESHOLD):
                    state = "UP"
                    beat_count += 1
                    flash = True

                    now = time.time()
                    # beat BPM (per down-up bounce)
                    if last_beat_time:
                        delta = now - last_beat_time
                        if 0.3 < delta < 2.5:
                            bpm = 60.0 / delta
                    last_beat_time = now

                    # advance within the bar
                    beat_in_bar += 1
                    if beat_in_bar > 4:
                        beat_in_bar = 1
                        # we just completed a full 4/4 pattern (Down-Left-Right-Up)
                        bar_count += 1
                        if last_bar_time:
                            bar_delta = now - last_bar_time
                            if bar_delta > 0:
                                bar_bpm = 60.0 / bar_delta
                        last_bar_time = now

            elif state == "UP":
                # after rebound, once high enough, arm for next down-stroke
                if curr_y < (lowest_y - THRESHOLD / 2):
                    state = "DOWN"
                    lowest_y = curr_y

            # threshold line
            if state == "DOWN":
                trigger_y = int((lowest_y - THRESHOLD) * h)
                if 0 <= trigger_y < h:
                    cv2.line(img, (0, trigger_y), (w, trigger_y), (0, 255, 255), 2)
                    cv2.putText(img, "Pull UP past line",
                                (10, trigger_y - 10), font, 0.7, (0, 255, 255), 2)

        else:
            state = "WAITING"
            lowest_y = 0.0

        # ---- Visuals ----
        if flash_timer > 0:
            flash_timer -= 1
        if flash:
            flash_timer = 5

        if flash_timer > 0:
            cv2.circle(img, (50, 50), 40, (0, 255, 0), -1)

        # 4/4 pattern around index finger
        if index_px is not None:
            draw_44_pattern(img, index_px, index_py, beat_in_bar)

        # HUD
        state_color = (0, 255, 0) if state == "UP" else (0, 0, 255)
        cv2.putText(img, f"State: {state}", (30, 80), font, 1, state_color, 2)
        cv2.putText(img, f"Beat BPM: {int(bpm)}", (30, 130), font, 1.2, (255, 255, 255), 3)
        cv2.putText(img, f"Bar BPM (1 cycle): {int(bar_bpm)}", (30, 170), font, 1.0, (0, 255, 255), 2)
        cv2.putText(img, f"Total beats: {beat_count}  |  Total bars: {bar_count}",
                    (30, 210), font, 0.9, (200, 200, 200), 2)
        cv2.putText(img, PATTERN_TEXT[beat_in_bar], (30, 250), font, 0.9, (255, 255, 0), 2)
        cv2.putText(img, "Do: Down, Left, Right, Up = 1 full pattern (1 bar).",
                    (30, h - 60), font, 0.8, (200, 200, 200), 2)
        cv2.putText(img, "Each down-up bounce = 1 beat; 4 beats = 1 bar.",
                    (30, h - 30), font, 0.8, (200, 200, 200), 2)

        cv2.imshow("Sticky 4/4 Conductor", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
