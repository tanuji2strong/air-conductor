import cv2
import mediapipe as mp
import time
from collections import deque
import numpy as np

# --- CONFIGURATION ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SMOOTHING_WINDOW = 5       # How many frames to average (stabilizes shaky hands)
MIN_MOVEMENT_THRESH = 0.02 # Minimum vertical movement required to register a beat (0.0 to 1.0)
TIMEOUT_SECONDS = 2.0      # Reset BPM if no motion for 2 seconds

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class BeatDetector:
    def __init__(self):
        self.y_history = deque(maxlen=SMOOTHING_WINDOW)
        self.last_beat_time = 0
        self.bpm = 0
        self.previous_y = 0
        self.direction = 0 # 1 = moving down, -1 = moving up
        self.last_valley_y = 0
        
        # UI visualizers
        self.flash_timer = 0
        
    def update(self, raw_y):
        """
        Input: raw_y is the normalized Y position (0.0 top, 1.0 bottom)
        Returns: current_bpm, is_beat_frame (True/False)
        """
        # 1. Smooth the data (Moving Average)
        self.y_history.append(raw_y)
        smoothed_y = sum(self.y_history) / len(self.y_history)
        
        current_time = time.time()
        is_beat = False
        
        # 2. Determine Direction
        # In OpenCV: Increase Y = Moving DOWN, Decrease Y = Moving UP
        if smoothed_y > self.previous_y:
            current_direction = 1 # Down
        else:
            current_direction = -1 # Up

        # 3. Detect "Valley" (The Ictus)
        # If we were moving DOWN (1), and now we are moving UP (-1), that's the bottom.
        if self.direction == 1 and current_direction == -1:
            
            # Check magnitude (did they move enough? or is it just jitter?)
            # We compare current bottom against the point where we started going down? 
            # Simplified: just ensure the y value is reasonably low or changed enough.
            
            # Calculate Time Delta
            delta_t = current_time - self.last_beat_time
            
            # Filter unrealistic speeds (Conducting is usually 30 to 200 BPM)
            # 0.3s = 200 BPM, 2.0s = 30 BPM
            if 0.3 < delta_t < 2.0:
                instant_bpm = 60.0 / delta_t
                # Smooth the BPM changes slightly
                if self.bpm == 0:
                    self.bpm = instant_bpm
                else:
                    self.bpm = (self.bpm * 0.6) + (instant_bpm * 0.4)
                
                self.last_beat_time = current_time
                is_beat = True
                self.flash_timer = 5 # Flash for 5 frames

        # Update state
        self.previous_y = smoothed_y
        self.direction = current_direction
        
        # Timeout reset
        if current_time - self.last_beat_time > TIMEOUT_SECONDS:
            self.bpm = 0
            
        return self.bpm, is_beat

# --- MAIN LOOP ---
def main():
    cap = cv2.VideoCapture(0)
    detector = BeatDetector()
    
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            ret, img = cap.read()
            if not ret: break
            
            # Setup Image
            img = cv2.resize(img, (CAMERA_WIDTH, CAMERA_HEIGHT))
            img = cv2.flip(img, 1) 
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            bpm_text = "0"
            is_beat = False
            
            if results.multi_hand_landmarks:
                # We only care about the first hand detected for conducting
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # TRACK WRIST (Landmark 0)
                wrist_y = hand_landmarks.landmark[0].y # Normalized 0.0 to 1.0
                
                # Update Detector
                bpm_val, is_beat = detector.update(wrist_y)
                bpm_text = f"{int(bpm_val)}"

                # Visual Debugging: Draw the wrist level
                h, w, c = img.shape
                cy = int(wrist_y * h)
                cv2.line(img, (0, cy), (w, cy), (255, 0, 0), 1)

            # --- UI FEEDBACK ---
            # 1. Flash effect on beat
            if detector.flash_timer > 0:
                cv2.circle(img, (50, 50), 30, (0, 255, 255), -1) # Yellow Flash
                detector.flash_timer -= 1
            else:
                cv2.circle(img, (50, 50), 30, (50, 50, 50), -1) # Gray
            
            # 2. Display BPM
            cv2.putText(img, f"BPM: {bpm_text}", (100, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            cv2.imshow('Beat Detector', img)
            if cv2.waitKey(5) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
