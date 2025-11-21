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
beat_count = 0
beat_times = []

# movement tracking
prev_x = None
prev_y = None
moving = False
total_dist = 0.0
still_frames = 0

# thresholds (tune if needed)
STEP_MOVE_THRESH = 0.005      # per-frame movement to be considered "moving"
MIN_STROKE_DIST = 0.05        # total distance to accept as a stroke
STILL_FRAMES_N = 3            # how many consecutive still frames = stroke ended

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

        # draw landmarks
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # use first hand
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # wrist position (normalized 0~1)
            wrist = hand_landmarks.landmark[0]
            x = wrist.x
            y = wrist.y

            # movement logic
            if prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy = y - prev_y
                step_dist = math.sqrt(dx*dx + dy*dy)

                # if hand moved enough in this frame
                if step_dist > STEP_MOVE_THRESH:
                    moving = True
                    total_dist += step_dist
                    still_frames = 0
                else:
                    # not moving much
                    if moving:
                        still_frames += 1
                        # if we've been still for several frames and stroke was long enough
                        if still_frames >= STILL_FRAMES_N and total_dist > MIN_STROKE_DIST:
                            beat_count += 1
                            beat_times.append(time.time())
                            # reset stroke info
                            moving = False
                            total_dist = 0.0
                    else:
                        # already not moving, do nothing
                        pass

            prev_x, prev_y = x, y

        else:
            cv2.putText(img, "Show ONE hand", (30, 80),
                        fontFace, 0.8, (0, 0, 255), 2, lineType)
            # reset tracking when no hand
            prev_x = prev_y = None
            moving = False
            total_dist = 0.0
            still_frames = 0

        bpm = compute_bpm(beat_times)

        # display info
        cv2.putText(img, f"Beats: {beat_count}", (30, 140),
                    fontFace, 1.2, (255, 255, 255), 3, lineType)
        cv2.putText(img, f"BPM: {int(bpm)}", (30, 200),
                    fontFace, 1.2, (0, 255, 0), 3, lineType)

        # optional debug: show if currently moving
        status_text = "Moving" if moving else "Still"
        cv2.putText(img, status_text, (30, 260),
                    fontFace, 0.9, (255, 255, 0), 2, lineType)

        cv2.imshow("Stroke BPM (One Hand)", img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
