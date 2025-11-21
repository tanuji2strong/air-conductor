import cv2
import mediapipe as mp
import time
from collections import deque

# --- CONFIGURATION ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SMOOTHING_WINDOW = 5       

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class ConductorBrain:
    def __init__(self):
        # Y-Axis (Tempo) Data
        self.y_history = deque(maxlen=SMOOTHING_WINDOW)
        self.last_beat_time = 0
        self.bpm = 0
        self.previous_y = 0
        self.direction = 0 
        self.flash_timer = 0
        
        # X-Axis (Section) Data
        self.current_section = "CENTER" # LEFT, CENTER, RIGHT

    def update(self, x_norm, y_norm):
        """
        x_norm, y_norm: Normalized coordinates (0.0 to 1.0)
        """
        # --- 1. CALCULATE BEAT (Y-AXIS) ---
        self.y_history.append(y_norm)
        smoothed_y = sum(self.y_history) / len(self.y_history)
        
        is_beat = False
        current_time = time.time()
        
        # Check Direction (Down vs Up)
        if smoothed_y > self.previous_y:
            current_direction = 1 # Down
        else:
            current_direction = -1 # Up

        # Detect Valley (The "Ictus")
        if self.direction == 1 and current_direction == -1:
            delta_t = current_time - self.last_beat_time
            if 0.3 < delta_t < 2.0:
                instant_bpm = 60.0 / delta_t
                # Smooth BPM
                if self.bpm == 0: self.bpm = instant_bpm
                else: self.bpm = (self.bpm * 0.7) + (instant_bpm * 0.3)
                
                self.last_beat_time = current_time
                is_beat = True
                self.flash_timer = 5 

        self.previous_y = smoothed_y
        self.direction = current_direction
        
        # Timeout (Reset if stopped)
        if current_time - self.last_beat_time > 2.0:
            self.bpm = 0

        # --- 2. CALCULATE SECTION (X-AXIS) ---
        # Remember: In OpenCV mirror view, 0 is Left side of SCREEN (User's Right hand)
        # Let's keep it simple based on Screen Coordinates
        if x_norm < 0.33:
            self.current_section = "LEFT (Bass/Drums)"
        elif x_norm > 0.66:
            self.current_section = "RIGHT (Violins/Flute)"
        else:
            self.current_section = "CENTER (All)"

        return self.bpm, is_beat, self.current_section

# --- VISUAL DRAWING HELPERS ---
def draw_zones(img, active_section):
    h, w, c = img.shape
    
    # Colors (BGR)
    c_active = (0, 255, 0)   # Green
    c_inactive = (50, 50, 50) # Dark Gray
    
    # Zone 1: Left
    color = c_active if "LEFT" in active_section else c_inactive
    cv2.rectangle(img, (0, h-50), (int(w/3), h), color, -1)
    cv2.putText(img, "LOW", (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Zone 2: Center
    color = c_active if "CENTER" in active_section else c_inactive
    cv2.rectangle(img, (int(w/3), h-50), (int(2*w/3), h), color, -1)
    cv2.putText(img, "ALL", (int(w/3)+50, h-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Zone 3: Right
    color = c_active if "RIGHT" in active_section else c_inactive
    cv2.rectangle(img, (int(2*w/3), h-50), (w, h), color, -1)
    cv2.putText(img, "HIGH", (int(2*w/3)+50, h-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Draw Divider Lines
    cv2.line(img, (int(w/3), 0), (int(w/3), h), (100,100,100), 1)
    cv2.line(img, (int(2*w/3), 0), (int(2*w/3), h), (100,100,100), 1)

# --- MAIN LOOP ---
def main():
    cap = cv2.VideoCapture(0)
    brain = ConductorBrain()
    
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret: break
            
            img = cv2.resize(img, (CAMERA_WIDTH, CAMERA_HEIGHT))
            img = cv2.flip(img, 1) # Mirror view
            
            # Draw UI Zones Background
            draw_zones(img, brain.current_section)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                # Get the first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get Wrist Coordinates
                wrist = hand_landmarks.landmark[0]
                
                # Update Brain
                bpm, is_beat, section = brain.update(wrist.x, wrist.y)
                
                # Visual Feedback
                # 1. Beat Flash
                if brain.flash_timer > 0:
                    cv2.circle(img, (50, 50), 30, (0, 255, 255), -1)
                    brain.flash_timer -= 1
                
                # 2. BPM Text
                cv2.putText(img, f"BPM: {int(bpm)}", (100, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # 3. Section Text
                cv2.putText(img, f"Section: {section}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

            cv2.imshow('Air Conductor Pro', img)
            if cv2.waitKey(5) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
