import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
lineType = cv2.LINE_AA

# ===== BPM STATE =====
prev_state = "center"
beat_count = 0
beat_times = []
last_beat_time = None
MIN_BEAT_INTERVAL = 0.25

LEFT_THRESHOLD = -0.15
RIGHT_THRESHOLD = 0.15


def compute_bpm(beat_times, window=6):
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


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
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

        # === MIRROR CAMERA ===
        img = cv2.flip(img, 1)

        img = cv2.resize(img, (w, h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hl, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # ===== LEFT/RIGHT DETECTION =====
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            wrist_xs = []

            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                wrist_xs.append(wrist.x)

            avg_x = (sum(wrist_xs) / len(wrist_xs)) - 0.5

            if avg_x < LEFT_THRESHOLD:
                state = "left"
            elif avg_x > RIGHT_THRESHOLD:
                state = "right"
            else:
                state = "center"

            now = time.time()
            is_transition = (
                (prev_state == "left" and state == "right") or
                (prev_state == "right" and state == "left")
            )

            if is_transition:
                if last_beat_time is None or (now - last_beat_time) > MIN_BEAT_INTERVAL:
                    beat_count += 1
                    beat_times.append(now)
                    last_beat_time = now

            prev_state = state

            cv2.putText(img, f"State: {state}", (30, 80),
                        fontFace, 1, (0, 255, 255), 2, lineType)
        else:
            cv2.putText(img, "Show BOTH hands to count BPM", (30, 80),
                        fontFace, 0.8, (0, 0, 255), 2, lineType)

        # BPM
        bpm = compute_bpm(beat_times)

        cv2.putText(img, f"Beats: {beat_count}", (30, 140),
                    fontFace, 1.2, (255, 255, 255), 3, lineType)
        cv2.putText(img, f"BPM: {int(bpm)}", (30, 200),
                    fontFace, 1.2, (0, 255, 0), 3, lineType)

        cv2.imshow("BPM Conductor (Mirrored)", img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
