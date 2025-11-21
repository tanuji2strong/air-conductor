import cv2
import mediapipe as mp
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
lineType = cv2.LINE_AA

# ===== BEAT / BPM STATE =====
beat_count = 0          # total beats detected
beat_times = []         # timestamps for BPM
beat_in_bar = 1         # 1,2,3,4 in a 4/4 bar

# movement tracking (one hand)
prev_x = None
prev_y = None
last_dir = "none"       # last detected direction
dir_frames = 0          # how many consecutive frames in same direction

# thresholds (tune as needed)
MOVE_THRESH = 0.02      # minimum movement per frame to consider "moving"
DIR_HOLD_FRAMES = 3     # frames in same direction to accept as a beat

# 4/4 conducting pattern: beat -> expected direction
PATTERN = {
    1: "down",
    2: "left",
    3: "right",
    4: "up"
}

def compute_bpm(beat_times, window=6):
    """Compute BPM from recent beat timestamps."""
    if len(beat_times) < 2:
        return 0.0

    recent = beat_times[-window:]
    intervals = []
    for i in range(1, len(recent)):
        intervals.append(recent[i] - recent[i - 1])

    if not intervals:
        return 0.0

    avg_interval = sum(intervals) / len(intervals)
    if avg_interval <= 0:
        return 0.0

    return 60.0 / avg_interval


def get_direction(dx, dy, move_thresh):
    """Return main direction: 'left', 'right', 'up', 'down', or 'none'."""
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < move_thresh:
        return "none"

    # vertical vs horizontal dominance
    if abs(dy) >= abs(dx):
        # vertical movement
        return "down" if dy > 0 else "up"
    else:
        # horizontal movement
        return "right" if dx > 0 else "left"


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
        if not ret:
            print("Cannot receive frame")
            break

        # mirror camera so it feels natural
        img = cv2.flip(img, 1)

        img = cv2.resize(img, (w, h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        current_dir = "none"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # wrist position (normalized 0~1)
            wrist = hand_landmarks.landmark[0]
            x = wrist.x
            y = wrist.y

            if prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy = y - prev_y

                current_dir = get_direction(dx, dy, MOVE_THRESH)

                # direction consistency counting
                if current_dir == "none":
                    # no meaningful movement
                    dir_frames = 0
                else:
                    if current_dir == last_dir:
                        dir_frames += 1
                    else:
                        last_dir = current_dir
                        dir_frames = 1

                    # check if this direction matches the expected one for this beat
                    expected_dir = PATTERN[beat_in_bar]

                    if current_dir == expected_dir and dir_frames == DIR_HOLD_FRAMES:
                        # counted as a beat in the 4/4 pattern
                        beat_count += 1
                        beat_times.append(time.time())

                        # advance beat in bar: 1 -> 2 -> 3 -> 4 -> 1
                        if beat_in_bar == 4:
                            beat_in_bar = 1
                        else:
                            beat_in_bar += 1

                        # prevent multiple triggers on same stroke
                        # (will need movement change again)
                        dir_frames = DIR_HOLD_FRAMES

            prev_x, prev_y = x, y

        else:
            cv2.putText(img, "Show ONE hand", (30, 80),
                        fontFace, 0.8, (0, 0, 255), 2, lineType)
            prev_x = None
            prev_y = None
            last_dir = "none"
            dir_frames = 0

        bpm = compute_bpm(beat_times)

        # ===== DISPLAY TEXT =====
        cv2.putText(img, f"Beats total: {beat_count}", (30, 140),
                    fontFace, 1.0, (255, 255, 255), 2, lineType)
        cv2.putText(img, f"BPM: {int(bpm)}", (30, 180),
                    fontFace, 1.0, (0, 255, 0), 2, lineType)

        expected_dir = PATTERN[beat_in_bar]
        cv2.putText(img, f"Beat in bar: {beat_in_bar} (expect: {expected_dir})",
                    (30, 220), fontFace, 0.8, (255, 255, 0), 2, lineType)

        cv2.putText(img, f"Current dir: {current_dir}", (30, 260),
                    fontFace, 0.8, (0, 255, 255), 2, lineType)

        cv2.imshow("4/4 Conductor (Down-Left-Right-Up)", img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
