import cv2
import mediapipe as mp
import time
from math import sqrt

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- CONFIG ---
COUNTDOWN_SECONDS = 3
RADIUS = 80             # radius (pixels) for target zones (bigger = easier)
MIN_INTERVAL = 0.25     # min seconds between beats (avoid spam)
MAX_INTERVAL = 3.0      # max seconds between beats (too slow -> ignore)

USE_RIGHT_ARM = False    # True: use right arm (12,14,16,20,22); False: use left  (reverse if mirror camera)


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

beat_count = 0
bar_count = 0
expected_beat = 1       # 1 -> 2 -> 3 -> 4 -> 1
last_beat_time = None
last_bar_time = None
bpm = 0.0
bar_bpm = 0.0
flash_timer = 0

PATTERN_TEXT = {
    1: "1: DOWN (center -> bottom)",
    2: "2: LEFT",
    3: "3: RIGHT",
    4: "4: UP (back to start)"
}

def dist(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def draw_pattern_and_targets(img, cx, cy, active_beat):
    """
    Draw fixed 4/4 cross in the center of the screen.
    Return positions of the 4 target points.
    """
    L = 120
    base_color = (60, 60, 60)
    active_color = (0, 255, 255)

    center = (cx, cy)
    bottom = (cx, cy + L)
    left   = (cx - L, cy)
    right  = (cx + L, cy)
    top    = (cx, cy - L)

    # Base cross
    cv2.line(img, (cx, cy - L), (cx, cy + L), base_color, 2)
    cv2.line(img, (cx - L, cy), (cx + L, cy), base_color, 2)

    # Active stroke highlight
    if active_beat == 1:
        cv2.line(img, center, bottom, active_color, 5)
        cv2.circle(img, bottom, 10, active_color, -1)
    elif active_beat == 2:
        cv2.line(img, center, left, active_color, 5)
        cv2.circle(img, left, 10, active_color, -1)
    elif active_beat == 3:
        cv2.line(img, center, right, active_color, 5)
        cv2.circle(img, right, 10, active_color, -1)
    elif active_beat == 4:
        cv2.line(img, center, top, active_color, 5)
        cv2.circle(img, top, 10, active_color, -1)

    # Labels
    cv2.putText(img, "1", (bottom[0] + 10, bottom[1]), font, 0.6, (150,150,150), 1)
    cv2.putText(img, "2", (left[0] - 10, left[1] - 10), font, 0.6, (150,150,150), 1)
    cv2.putText(img, "3", (right[0] - 10, right[1] - 10), font, 0.6, (150,150,150), 1)
    cv2.putText(img, "4", (top[0] + 10, top[1] - 10), font, 0.6, (150,150,150), 1)

    return center, bottom, left, right, top

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
            text = "START"
            color = (0, 255, 0)

        size, _ = cv2.getTextSize(text, font, 3, 6)
        tw, th = size
        cx, cy = w // 2, h // 2

        cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                    font, 3, color, 6)
        cv2.putText(frame, "Move your ARM: Down -> Left -> Right -> Up",
                    (30, h - 40), font, 0.7, (200, 200, 200), 2)

        cv2.imshow("4/4 Pose Conductor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        if elapsed >= COUNTDOWN_SECONDS + 1:
            break
    return True

with mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
) as pose:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    if not run_countdown():
        cap.release()
        cv2.destroyAllWindows()
        exit()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        h, w, _ = img.shape

        cx, cy = w // 2, h // 2

        # Draw pattern & get zone centers
        center_pt, bottom_pt, left_pt, right_pt, top_pt = draw_pattern_and_targets(
            img, cx, cy, expected_beat
        )

        # Pose processing
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        now = time.time()

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            if USE_RIGHT_ARM:
                shoulder_idx = 12
                elbow_idx    = 14
                wrist_idx    = 16
                index_idx    = 20
                thumb_idx    = 22
            else:
                shoulder_idx = 11
                elbow_idx    = 13
                wrist_idx    = 15
                index_idx    = 19
                thumb_idx    = 21

            # Extract coordinates (if visibility is decent)
            def get_xy(i):
                p = lm[i]
                return int(p.x * w), int(p.y * h), p.visibility

            sx, sy, s_vis = get_xy(shoulder_idx)
            ex, ey, e_vis = get_xy(elbow_idx)
            wx, wy, w_vis = get_xy(wrist_idx)
            ix, iy, i_vis = get_xy(index_idx)
            tx, ty, t_vis = get_xy(thumb_idx)

            # Draw the arm (shoulder->elbow->wrist->index)
            if s_vis > 0.4 and e_vis > 0.4:
                cv2.line(img, (sx, sy), (ex, ey), (255, 255, 0), 3)
            if e_vis > 0.4 and w_vis > 0.4:
                cv2.line(img, (ex, ey), (wx, wy), (0, 255, 255), 3)
            if w_vis > 0.4 and i_vis > 0.4:
                cv2.line(img, (wx, wy), (ix, iy), (0, 200, 255), 3)

            # Mark joints
            cv2.circle(img, (sx, sy), 6, (255, 255, 0), -1)
            cv2.circle(img, (ex, ey), 6, (255, 255, 0), -1)
            cv2.circle(img, (wx, wy), 6, (0, 255, 255), -1)
            cv2.circle(img, (ix, iy), 8, (0, 255, 0), -1)

            fingertip = (ix, iy)

            # Timing filter
            interval_ok = False
            if last_beat_time is None:
                interval_ok = True
            else:
                dt = now - last_beat_time
                if MIN_INTERVAL < dt < MAX_INTERVAL:
                    interval_ok = True

            if interval_ok:
                # Beat 1: go near bottom
                if expected_beat == 1 and dist(fingertip, bottom_pt) < RADIUS:
                    beat_count += 1
                    flash_timer = 5
                    if last_beat_time is not None:
                        bpm = 60.0 / (now - last_beat_time)
                    last_beat_time = now
                    expected_beat = 2

                # Beat 2: go near left
                elif expected_beat == 2 and dist(fingertip, left_pt) < RADIUS:
                    beat_count += 1
                    flash_timer = 5
                    if last_beat_time is not None:
                        bpm = 60.0 / (now - last_beat_time)
                    last_beat_time = now
                    expected_beat = 3

                # Beat 3: go near right
                elif expected_beat == 3 and dist(fingertip, right_pt) < RADIUS:
                    beat_count += 1
                    flash_timer = 5
                    if last_beat_time is not None:
                        bpm = 60.0 / (now - last_beat_time)
                    last_beat_time = now
                    expected_beat = 4

                # Beat 4: go near top
                elif expected_beat == 4 and dist(fingertip, top_pt) < RADIUS:
                    beat_count += 1
                    flash_timer = 5
                    if last_beat_time is not None:
                        bpm = 60.0 / (now - last_beat_time)
                    last_beat_time = now

                    # full bar done
                    expected_beat = 1
                    bar_count += 1

                    if last_bar_time is not None:
                        bar_dt = now - last_bar_time
                        if bar_dt > 0:
                            bar_bpm = 60.0 / bar_dt
                    else:
                        bar_bpm = 0.0
                    last_bar_time = now

        # Flash on beat
        if flash_timer > 0:
            cv2.circle(img, (w - 60, 60), 30, (0, 255, 0), -1)
            flash_timer -= 1

        # HUD
        cv2.putText(img, f"Beat BPM: {int(bpm)}", (20, 50), font, 1.0, (255,255,255), 2)
        cv2.putText(img, f"Total Beats: {beat_count}", (20, 90), font, 0.9, (200,200,200), 2)
        cv2.putText(img, f"Bars: {bar_count}  |  Bar BPM: {int(bar_bpm)}",
                    (20, 130), font, 0.9, (0,255,255), 2)
        cv2.putText(img, f"Next: {PATTERN_TEXT[expected_beat]}",
                    (20, h - 40), font, 0.9, (255,255,0), 2)

        cv2.imshow("4/4 Pose Conductor", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
