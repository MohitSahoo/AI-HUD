import cv2
import numpy as np
import time
import os
import pygame
from ultralytics import YOLO
from download_model import download_yolo_model
from collections import deque

# Initialize pygame for sound
pygame.mixer.init()

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n.pt')
ALERT_SOUND_PATH = os.path.join(BASE_DIR, 'utils', 'alert.wav')

# Constants for improved detection
CALIBRATION_FRAMES = 30  # Number of frames for calibration
CONFIDENCE_THRESHOLD = 0.25  # Base confidence threshold
DROWSY_CONSEC_FRAMES = 15  # Reduced frames to trigger alert (more sensitive)
ALERT_COOLDOWN = 3  # Reduced cooldown between alerts
BLINK_THRESHOLD = 0.3  # Seconds to distinguish blinks from drowsiness
CONFIDENCE_WINDOW = 10  # Number of frames to average confidence

# Drowsiness detection parameters - adjusted for better sensitivity
HEAD_NOD_THRESHOLD = 0.65  # Decreased threshold (more sensitive to head nodding)
FACE_TILT_THRESHOLD = 0.6  # Decreased threshold (more sensitive to face tilting)
MOVEMENT_THRESHOLD = 0.2  # Adjusted for better movement detection
EYE_CLOSURE_THRESHOLD = 0.6  # Adjusted for better eye closure detection

# YOLO class names (COCO dataset)
YOLO_CLASSES = {
    0: 'person',  # We'll use this for drowsiness detection
}

def try_camera_indices():
    """Try different camera indices to find an available camera."""
    for i in range(10):  # Try first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cap.release()
                return i
            cap.release()
    return None

def get_main_person(boxes, frame_shape):
    """
    Get the main person (largest/most prominent) from the detected boxes.
    Returns the box with the highest confidence and largest area.
    """
    if not boxes:
        return None
    
    main_box = None
    max_score = -1
    
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:  # person class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Calculate area and center position
            area = (x2 - x1) * (y2 - y1)
            center_y = (y1 + y2) / 2
            
            # Calculate score based on area, confidence, and position
            # Prefer boxes that are larger, more confident, and more centered
            position_score = 1 - abs(center_y - frame_shape[0]/2) / (frame_shape[0]/2)
            score = area * conf * (0.7 + 0.3 * position_score)  # Weighted score
            
            if score > max_score:
                max_score = score
                main_box = (x1, y1, x2, y2, conf)
    
    return main_box

class DrowsinessDetector:
    def __init__(self):
        self.last_positions = deque(maxlen=10)
        self.last_angles = deque(maxlen=10)
        self.movement_history = deque(maxlen=30)
        self.eye_state_history = deque(maxlen=5)
        self.calibration_data = {
            'normal_angle': None,
            'normal_position': None,
            'normal_movement': None
        }
        self.normal_movement_threshold = None
        self.normal_angle_threshold = None
        self.last_main_box = None  # Track the main person's box
    
    def calculate_face_angle(self, face_box):
        """Calculate the angle of the face based on box dimensions"""
        x1, y1, x2, y2 = face_box
        width = x2 - x1
        height = y2 - y1
        return width / height if height != 0 else 1.0
    
    def calculate_movement(self, current_pos, last_pos):
        """Calculate the amount of movement between positions"""
        if not last_pos:
            return 0
        return np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
    
    def is_drowsy(self, face_box, frame_shape):
        """
        Enhanced drowsiness detection using multiple criteria with improved sensitivity
        """
        x1, y1, x2, y2 = face_box
        height, width = frame_shape[:2]
        
        # Calculate face center and properties
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        face_width = x2 - x1
        face_height = y2 - y1
        current_angle = self.calculate_face_angle(face_box)
        
        # Store current position and angle
        self.last_positions.append((face_center_x, face_center_y))
        self.last_angles.append(current_angle)
        
        # Calculate movement
        if len(self.last_positions) > 1:
            movement = self.calculate_movement(
                self.last_positions[-1],
                self.last_positions[-2]
            )
            self.movement_history.append(movement)
        
        # Initialize drowsiness score
        drowsiness_score = 0
        drowsiness_factors = []
        
        # 1. Check for head nodding - more sensitive
        if len(self.last_positions) > 3:
            recent_positions = list(self.last_positions)[-3:]
            avg_y = np.mean([p[1] for p in recent_positions])
            if avg_y > height * HEAD_NOD_THRESHOLD:
                # Count if head is low for a short period
                if any(p[1] > height * HEAD_NOD_THRESHOLD for p in recent_positions[-2:]):
                    drowsiness_score += 1
                    drowsiness_factors.append("head_nodding")
        
        # 2. Check for face tilting - more sensitive
        if len(self.last_angles) > 3:
            recent_angles = list(self.last_angles)[-3:]
            angle_variation = np.std(recent_angles)
            if angle_variation > FACE_TILT_THRESHOLD:
                # Count if tilt is significant
                if any(abs(a - np.mean(recent_angles)) > FACE_TILT_THRESHOLD/2 for a in recent_angles[-2:]):
                    drowsiness_score += 1
                    drowsiness_factors.append("face_tilting")
        
        # 3. Check for reduced movement - more sensitive
        if len(self.movement_history) > 5:
            recent_movement = list(self.movement_history)[-5:]
            avg_movement = np.mean(recent_movement)
            if avg_movement < MOVEMENT_THRESHOLD:
                # Count if movement is low
                if any(m < MOVEMENT_THRESHOLD * 1.2 for m in recent_movement[-3:]):
                    drowsiness_score += 1
                    drowsiness_factors.append("reduced_movement")
        
        # 4. Check for eye closure - more sensitive
        if len(self.eye_state_history) > 2:
            recent_eye_states = list(self.eye_state_history)[-2:]
            if face_height < height * EYE_CLOSURE_THRESHOLD:
                # Count if eye closure is detected
                if any(state > 0 for state in recent_eye_states):
                    drowsiness_score += 1
                    drowsiness_factors.append("possible_eye_closure")
        
        # Store eye state
        self.eye_state_history.append(drowsiness_score)
        
        # Determine if drowsy based on multiple factors - more sensitive
        # Need at least 2 factors to trigger, and they must be different types
        is_drowsy = drowsiness_score >= 2 and len(set(drowsiness_factors)) >= 2
        
        return is_drowsy, drowsiness_factors, drowsiness_score

def calibrate_detection(model, cap, num_frames=CALIBRATION_FRAMES):
    """
    Calibrate the detection system by analyzing normal face positions and movements.
    """
    print("Calibrating detection system... Please look straight ahead and move normally.")
    detector = DrowsinessDetector()
    face_positions = []
    face_angles = []
    movements = []
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
            
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_positions.append((x1, y1, x2, y2))
                    face_angles.append(detector.calculate_face_angle((x1, y1, x2, y2)))
    
    if face_positions:
        print("Calibration complete. Starting drowsiness detection.")
        return detector
    else:
        print("Calibration failed. Using default settings.")
        return DrowsinessDetector()

def main():
    # Check if we have a trained model, otherwise use the base model
    if os.path.exists('runs/detect/train/weights/best.pt'):
        model_path = 'runs/detect/train/weights/best.pt'
        print(f"Using trained model: {model_path}")
    else:
        # Download or use existing base model
        model_path = download_yolo_model()
        if model_path is None:
            print("Error: Could not download or find the base YOLOv8 model.")
            return
        print(f"Trained model not found. Using base model: {model_path}")
        print("Note: For best results, train the model first using train_yolo_model.py")
    
    # Initialize YOLO model
    model = YOLO(model_path)
    
    # Load alert sound
    alert_sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
    
    # Try to open camera
    camera_index = try_camera_indices()
    if camera_index is None:
        print("Error: Could not find an available camera")
        return
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Calibrate detection
    detector = calibrate_detection(model, cap)
    
    # Initialize detection variables
    COUNTER = 0
    ALARM_ON = False
    last_alert_time = 0
    blink_counter = 0
    last_blink_time = 0
    confidence_buffer = deque(maxlen=CONFIDENCE_WINDOW)
    
    print("Starting drowsiness detection. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break
            
        # Detect objects in the frame
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        # Get the main person from the frame
        main_person = None
        for result in results:
            main_person = get_main_person(result.boxes, frame.shape)
            break
        
        # Process detection
        drowsy_detected = False
        drowsy_boxes = []
        awake_boxes = []
        current_confidence = 0
        drowsiness_factors = []
        drowsiness_score = 0
        
        if main_person:
            x1, y1, x2, y2, conf = main_person
            is_drowsy, factors, score = detector.is_drowsy((x1, y1, x2, y2), frame.shape)
            
            if is_drowsy:
                drowsy_detected = True
                current_confidence = conf
                drowsy_boxes.append((x1, y1, x2, y2, conf, factors, score))
                drowsiness_factors.extend(factors)
                drowsiness_score = max(drowsiness_score, score)
            else:
                awake_boxes.append((x1, y1, x2, y2, conf))
        
        # Update confidence buffer
        if drowsy_detected:
            confidence_buffer.append(current_confidence)
        
        # Calculate average confidence
        avg_confidence = np.mean(confidence_buffer) if confidence_buffer else 0
        
        # Update drowsiness counter with improved logic
        current_time = time.time()
        if drowsy_detected and avg_confidence > CONFIDENCE_THRESHOLD:
            COUNTER += 1
            
            # Check for blinks (quick changes in drowsiness state)
            if current_time - last_blink_time > BLINK_THRESHOLD:
                blink_counter += 1
                last_blink_time = current_time
            
            # Trigger alert if conditions are met
            if COUNTER >= DROWSY_CONSEC_FRAMES and current_time - last_alert_time > ALERT_COOLDOWN:
                if not ALARM_ON:
                    ALARM_ON = True
                    print("DROWSINESS ALERT!")
                    alert_sound.play()
                    last_alert_time = current_time
                
                # Draw drowsiness alert with detailed information
                alert_text = f"DROWSINESS ALERT! (Score: {drowsiness_score})"
                cv2.putText(frame, alert_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show detected factors
                for i, factor in enumerate(set(drowsiness_factors)):
                    cv2.putText(frame, f"- {factor}", (10, 60 + i*30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            COUNTER = max(0, COUNTER - 1)  # Gradual decrease
            ALARM_ON = False
        
        # Draw all detections with improved visualization
        for x1, y1, x2, y2, conf, factors, score in drowsy_boxes:
            # Draw bounding box with score-based color intensity
            color_intensity = int(255 * (score / 4))  # Normalize by max score
            color = (0, 0, color_intensity)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence and score
            label = f"drowsy (score: {score})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for x1, y1, x2, y2, conf in awake_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"awake {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display status information
        cv2.putText(frame, f"Blink Count: {blink_counter}", (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Avg Confidence: {avg_confidence:.2f}", (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Drowsiness Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 