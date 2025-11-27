import cv2
import mediapipe as mp
import time
import collections
import numpy as np
import pygame

# --- AUDIO SYSTEM (Procedural Sound Generation) ---
class SoundEngine:
    def __init__(self, frequency=440, duration=0.1):
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        self.sound = self._generate_sine_wave(frequency, duration)
        
        # distinct sound for the "Down" beat (Beat 1) - Higher pitch
        self.high_sound = self._generate_sine_wave(880, duration)

    def _generate_sine_wave(self, freq, duration, sample_rate=44100):
        # Generate a sine wave array
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(freq * t * 2 * np.pi)
        # Normalize to 16-bit range
        audio = (tone * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(audio)

    def play_beat(self, beat_number):
        if beat_number == 1:
            self.high_sound.play()
        else:
            self.sound.play()

# --- BPM SMOOTHER ---
class BpmCalculator:
    def __init__(self, window_size=4):
        self.intervals = collections.deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update(self):
        now = time.time()
        delta = now - self.last_time
        self.last_time = now
        
        # Ignore intervals that are too long (e.g., if user paused for 3 seconds)
        if delta < 2.0: 
            self.intervals.append(delta)
            
        # Calculate Average
        if len(self.intervals) > 0:
            avg_interval = sum(self.intervals) / len(self.intervals)
            if avg_interval > 0:
                return 60.0 / avg_interval
        return 0.0

    def reset(self):
        self.intervals.clear()
        self.last_time = time.time()

# --- CONFIG ---
mp_pose = mp.solutions.pose
font = cv2.FONT_HERSHEY_SIMPLEX

VERTICAL_THRESH = 60    
HORIZONTAL_THRESH = 60  
USE_RIGHT_ARM = False   
TRAIL_LENGTH = 15      

# Initialize Systems
sound_engine = SoundEngine()
bpm_calc = BpmCalculator(window_size=4) # Average over last 4 beats

cap = cv2.VideoCapture(0)

# State tracking
beat_count = 0
bar_count = 0
expected_beat = 1 
trail_points = collections.deque(maxlen=TRAIL_LENGTH)

# Visual Flash state
flash_timer = 0

def draw_dynamic_overlay(img, sx, sy, active_beat, dx, dy):
    inactive_col = (60, 60, 60)
    active_col = (0, 255, 255) # Yellow
    
    # Crosshair
    cv2.line(img, (sx, sy - 150), (sx, sy + 150), inactive_col, 2)
    cv2.line(img, (sx - 150, sy), (sx + 150, sy), inactive_col, 2)

    # Next Beat Hints
    hint_offset = 100
    if active_beat == 1: # Expecting DOWN
        cv2.arrowedLine(img, (sx, sy), (sx, sy + hint_offset), active_col, 4)
        cv2.putText(img, "1: DOWN", (sx - 40, sy + hint_offset + 30), font, 0.8, active_col, 2)
    elif active_beat == 2: # Expecting LEFT (Inward)
        cv2.arrowedLine(img, (sx, sy + 50), (sx - hint_offset, sy + 50), active_col, 4)
        cv2.putText(img, "2: IN", (sx - hint_offset - 60, sy + 50), font, 0.8, active_col, 2)
    elif active_beat == 3: # Expecting RIGHT (Outward)
        cv2.arrowedLine(img, (sx, sy + 50), (sx + hint_offset, sy + 50), active_col, 4)
        cv2.putText(img, "3: OUT", (sx + hint_offset + 10, sy + 50), font, 0.8, active_col, 2)
    elif active_beat == 4: # Expecting UP
        cv2.arrowedLine(img, (sx, sy), (sx, sy - hint_offset), active_col, 4)
        cv2.putText(img, "4: UP", (sx - 20, sy - hint_offset - 10), font, 0.8, active_col, 2)

# --- MAIN LOOP ---
current_bpm = 0.0

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:

    while cap.isOpened():
        ret, img = cap.read()
        if not ret: break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Flash Effect (decays every frame)
        if flash_timer > 0:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            alpha = flash_timer / 10.0
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            flash_timer -= 1
        
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        sx, sy = 0, 0 
        dx, dy = 0, 0 

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

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

            sx, sy = int(shoulder.x * w), int(shoulder.y * h)
            ix, iy = int(index.x * w), int(index.y * h)
            
            # Distance logic
            dx = ix - sx
            dy = iy - sy

            # Draw Arm
            cv2.line(img, (sx,sy), (int(elbow.x*w), int(elbow.y*h)), (200,200,200), 2)
            cv2.line(img, (int(elbow.x*w), int(elbow.y*h)), (int(wrist.x*w), int(wrist.y*h)), (200,200,200), 2)
            
            # Draw Trail
            trail_points.append((ix, iy))
            for i in range(1, len(trail_points)):
                thickness = int((i / len(trail_points)) * 5 + 1) 
                cv2.line(img, trail_points[i-1], trail_points[i], (0, 255, 255), thickness)
            
            cv2.circle(img, (ix, iy), 10, (0, 0, 255), -1) 

            # --- LOGIC ---
            # Using simple debounce via time check in BpmCalculator could work, 
            # but we keep explicit logic here for threshold triggers
            now = time.time()
            time_since_last = now - bpm_calc.last_time
            
            triggered = False

            # Add a small refractory period (debounce) of 0.25s
            if time_since_last > 0.25: 
                
                if expected_beat == 1 and dy > VERTICAL_THRESH:
                    triggered = True
                elif expected_beat == 2 and dx < -HORIZONTAL_THRESH:
                    triggered = True
                elif expected_beat == 3 and dx > HORIZONTAL_THRESH:
                    triggered = True
                elif expected_beat == 4 and dy < -VERTICAL_THRESH:
                    triggered = True

                if triggered:
                    # 1. Play Sound
                    sound_engine.play_beat(expected_beat)
                    
                    # 2. Update BPM
                    current_bpm = bpm_calc.update()
                    
                    # 3. Visuals & Counters
                    beat_count += 1
                    flash_timer = 5 # Flash for 5 frames
                    
                    # 4. Cycle Beat
                    expected_beat += 1
                    if expected_beat > 4:
                        expected_beat = 1
                        bar_count += 1

        # Draw UI
        if sx != 0:
            draw_dynamic_overlay(img, sx, sy, expected_beat, dx, dy)

        # HUD
        # Color code BPM: Green is "Andante/Moderato" range, Red is fast/slow
        bpm_col = (0, 255, 0)
        if current_bpm > 140 or current_bpm < 40: bpm_col = (0, 0, 255)
        
        cv2.putText(img, f"BPM: {int(current_bpm)}", (30, 60), font, 1.5, bpm_col, 3)
        cv2.putText(img, f"Bar: {bar_count} | Beat: {beat_count}", (30, 100), font, 0.8, (220, 220, 220), 1)

        cv2.imshow("Dynamic Pose Conductor v2", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
