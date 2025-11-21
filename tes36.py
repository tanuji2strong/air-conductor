import cv2
import mediapipe as mp
import time
import collections

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- CONFIG ---
# Thresholds are relative to your shoulder position
# Higher number = you must make larger movements
VERTICAL_THRESH = 60    # Pixels up/down to trigger
HORIZONTAL_THRESH = 60  # Pixels left/right to trigger

USE_RIGHT_ARM = False    # True: Use Right Arm | False: Use Left Arm
TRAIL_LENGTH = 15       # Length of the visual trail

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# State tracking
beat_count = 0
bar_count = 0
expected_beat = 1  # 1 -> 2 -> 3 -> 4 -> 1
last_beat_time = time.time()
bpm = 0.0

# Store recent fingertip positions for the visual trail
trail_points = collections.deque(maxlen=TRAIL_LENGTH)

def draw_dynamic_overlay(img, sx, sy, active_beat, dx, dy):
    """
    Draws a cross centered on the SHOULDER (sx, sy).
    Highlights the quadrant you need to swipe into.
    """
    h, w, _ = img.shape
    
    # Draw the dynamic center (Shoulder anchor)
    cv2.circle(img, (sx, sy), 5, (100, 100, 100), -1)
    
    # Colors
    inactive_col = (60, 60, 60)
    active_col = (0, 255, 255) # Yellow
    
    # Draw the Crosshair (The "Thresholds")
    # Vertical Line
    cv2.line(img, (sx, sy - 150), (sx, sy + 150), inactive_col, 2)
    # Horizontal Line
    cv2.line(img, (sx - 150, sy), (sx + 150, sy), inactive_col, 2)

    # Visual hints for next beat
    hint_offset = 100
    if active_beat == 1: # Expecting DOWN
        cv2.arrowedLine(img, (sx, sy), (sx, sy + hint_offset), active_col, 4)
        cv2.putText(img, "DOWN", (sx - 20, sy + hint_offset + 20), font, 0.7, active_col, 2)
    elif active_beat == 2: # Expecting LEFT (Inward)
        cv2.arrowedLine(img, (sx, sy + 50), (sx - hint_offset, sy + 50), active_col, 4)
        cv2.putText(img, "LEFT", (sx - hint_offset - 50, sy + 50), font, 0.7, active_col, 2)
    elif active_beat == 3: # Expecting RIGHT (Outward)
        cv2.arrowedLine(img, (sx, sy + 50), (sx + hint_offset, sy + 50), active_col, 4)
        cv2.putText(img, "RIGHT", (sx + hint_offset + 10, sy + 50), font, 0.7, active_col, 2)
    elif active_beat == 4: # Expecting UP
        cv2.arrowedLine(img, (sx, sy), (sx, sy - hint_offset), active_col, 4)
        cv2.putText(img, "UP", (sx - 20, sy - hint_offset - 10), font, 0.7, active_col, 2)

    # Debug: Show current hand delta relative to shoulder
    # cv2.putText(img, f"dx:{dx} dy:{dy}", (w - 150, h - 20), font, 0.5, (100,255,100), 1)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:

    while cap.isOpened():
        ret, img = cap.read()
        if not ret: break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Convert to RGB
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        sx, sy = 0, 0 # Shoulder X, Y
        dx, dy = 0, 0 # Delta (Distance from hand to shoulder)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Select Left or Right arm landmarks
            if USE_RIGHT_ARM:
                shoulder = lm[12]
                elbow = lm[14]
                wrist = lm[16]
                index = lm[20]
            else:
                shoulder = lm[11]
                elbow = lm[13]
                wrist = lm[15]
                index = lm[19]

            # Get Pixel Coordinates
            sx, sy = int(shoulder.x * w), int(shoulder.y * h)
            ix, iy = int(index.x * w), int(index.y * h)
            
            # Calculate Delta (Hand position relative to shoulder)
            # dx > 0 means hand is to the RIGHT of shoulder
            # dy > 0 means hand is BELOW shoulder
            dx = ix - sx
            dy = iy - sy

            # --- DRAWING ---
            # 1. Draw Arm Skeleton
            cv2.line(img, (sx,sy), (int(elbow.x*w), int(elbow.y*h)), (255,255,255), 2)
            cv2.line(img, (int(elbow.x*w), int(elbow.y*h)), (int(wrist.x*w), int(wrist.y*h)), (255,255,255), 2)
            
            # 2. Update and Draw Trail
            trail_points.append((ix, iy))
            for i in range(1, len(trail_points)):
                # Fade out trace
                thickness = int(np_interp := (i / len(trail_points)) * 5 + 1) 
                cv2.line(img, trail_points[i-1], trail_points[i], (0, 255, 255), thickness)
            
            cv2.circle(img, (ix, iy), 10, (0, 0, 255), -1) # Fingertip

            # --- LOGIC: THRESHOLD CROSSING ---
            # We add a simple "Cooldown" to prevent double counting quickly
            now = time.time()
            time_since_last = now - last_beat_time
            
            triggered = False

            if time_since_last > 0.25: # Debounce
                
                # BEAT 1: Hand must be significantly BELOW shoulder
                if expected_beat == 1:
                    if dy > VERTICAL_THRESH: 
                        triggered = True

                # BEAT 2: Hand must be significantly LEFT of shoulder
                # (Note: We usually keep hand somewhat low for beats 2 & 3, so we ignore Y check)
                elif expected_beat == 2:
                    if dx < -HORIZONTAL_THRESH:
                        triggered = True

                # BEAT 3: Hand must be significantly RIGHT of shoulder
                elif expected_beat == 3:
                    if dx > HORIZONTAL_THRESH:
                        triggered = True

                # BEAT 4: Hand must be significantly ABOVE shoulder
                elif expected_beat == 4:
                    if dy < -VERTICAL_THRESH:
                        triggered = True

                if triggered:
                    beat_count += 1
                    bpm = 60.0 / time_since_last if time_since_last > 0 else 0
                    last_beat_time = now
                    
                    # Cycle Beat
                    expected_beat += 1
                    if expected_beat > 4:
                        expected_beat = 1
                        bar_count += 1
                    
                    # Visual Flash trigger
                    cv2.circle(img, (ix, iy), 30, (0, 255, 0), 4)

        # Draw the UI overlay based on where the shoulder is
        if sx != 0:
            draw_dynamic_overlay(img, sx, sy, expected_beat, dx, dy)

        # HUD
        cv2.putText(img, f"BPM: {int(bpm)}", (30, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Bar: {bar_count} | Beat: {beat_count}", (30, 90), font, 0.8, (200, 200, 200), 1)

        cv2.imshow("Dynamic Pose Conductor", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
