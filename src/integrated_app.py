import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import threading
import pygame
from PIL import Image
import io
import base64
from drowsiness_detection import try_camera_indices
from download_model import download_yolo_model
import tempfile

# Initialize pygame for sound
pygame.mixer.init()

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n.pt')
ALERT_SOUND_PATH = os.path.join(BASE_DIR, 'utils', 'alert.wav')

class IntegratedSystem:
    def __init__(self):
        # Initialize models
        self.traffic_model = YOLO(MODEL_PATH)
        
        # Check if we have a trained model, otherwise use the base model
        if os.path.exists('runs/detect/train/weights/best.pt'):
            self.model_path = 'runs/detect/train/weights/best.pt'
        else:
            self.model_path = download_yolo_model()
        self.drowsiness_model = YOLO(self.model_path)
        
        # Load alert sound
        self.alert_sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
        
        # Initialize detection states
        self.DROWSY_CONSEC_FRAMES = 10  # Reduced for faster response
        self.drowsy_counter = 0
        self.ALARM_ON = False
        
        # Enhanced drowsiness detection parameters
        self.face_history = []  # Store recent face positions
        self.max_history = 15   # Increased history for better pattern detection
        self.movement_threshold = 0.015  # Reduced for more sensitive movement detection
        self.stability_threshold = 0.7   # Reduced for more sensitive stability detection
        self.confidence_threshold = 0.35  # Reduced for more sensitive detection
        
        # Drowsiness detection thresholds
        self.HEAD_TILT_THRESHOLD = 0.25    # Threshold for head tilt detection
        self.FACE_SIZE_THRESHOLD = 0.08    # Minimum face size ratio
        self.POSITION_THRESHOLD = 0.35     # Maximum allowed distance from center
        self.MOVEMENT_THRESHOLD = 0.015    # Threshold for significant movement
        self.DROWSY_SCORE_THRESHOLD = 0.4  # Reduced threshold for drowsiness score
        
        # Track eye closure and head movement patterns
        self.eye_state_history = []
        self.head_movement_history = []
        self.max_pattern_history = 5
        
        # Traffic detection classes
        self.traffic_classes = {
            0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 9: 'traffic light', 11: 'stop sign'
        }
        
        # Drowsiness classes - using eye states
        self.class_names = ['awake', 'drowsy']
        
        print("Drowsiness detection initialized with enhanced sensitivity")
        print("Model path:", self.model_path)

    def calculate_face_metrics(self, face_box, frame_shape):
        """Calculate detailed face metrics for drowsiness detection"""
        x1, y1, x2, y2 = face_box
        frame_height, frame_width = frame_shape[:2]
        
        # Calculate face center and properties
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Calculate face metrics
        face_area = face_width * face_height
        frame_area = frame_height * frame_width
        face_size_ratio = face_area / frame_area
        
        # Calculate position relative to frame center
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        center_distance = np.sqrt(
            ((face_center_x - frame_center_x) / frame_width) ** 2 +
            ((face_center_y - frame_center_y) / frame_height) ** 2
        )
        
        # Calculate aspect ratio and tilt
        aspect_ratio = face_width / face_height
        tilt_angle = abs(aspect_ratio - 0.75)  # 0.75 is ideal aspect ratio
        
        # Calculate vertical position (for detecting head nodding)
        vertical_position = face_center_y / frame_height
        
        return {
            'center': (face_center_x, face_center_y),
            'size_ratio': face_size_ratio,
            'center_distance': center_distance,
            'tilt_angle': tilt_angle,
            'aspect_ratio': aspect_ratio,
            'vertical_position': vertical_position,
            'width': face_width,
            'height': face_height
        }

    def detect_head_movement_pattern(self, current_metrics):
        """Detect patterns in head movement that indicate drowsiness"""
        if len(self.head_movement_history) < 3:
            return False
        
        # Get recent vertical positions
        recent_positions = [m['vertical_position'] for m in self.head_movement_history[-3:]]
        
        # Check for nodding pattern (up and down movement)
        position_diff = max(recent_positions) - min(recent_positions)
        if position_diff > 0.1:  # Significant vertical movement
            # Check if movement is slow (indicating drowsiness)
            if all(abs(recent_positions[i] - recent_positions[i-1]) < 0.05 for i in range(1, len(recent_positions))):
                return True
        
        return False

    def is_drowsy(self, face_box, frame_shape, confidence):
        """Enhanced drowsiness detection using multiple criteria"""
        if confidence < self.confidence_threshold:
            return False, 0.0
        
        # Calculate current face metrics
        current_metrics = self.calculate_face_metrics(face_box, frame_shape)
        
        # Update face history
        self.face_history.append(current_metrics)
        if len(self.face_history) > self.max_history:
            self.face_history.pop(0)
        
        # Update head movement history
        self.head_movement_history.append(current_metrics)
        if len(self.head_movement_history) > self.max_pattern_history:
            self.head_movement_history.pop(0)
        
        # Calculate drowsiness score based on multiple factors
        drowsiness_score = 0.0
        drowsiness_factors = []
        
        # 1. Check for head nodding (strong indicator of drowsiness)
        if self.detect_head_movement_pattern(current_metrics):
            drowsiness_score += 0.4
            drowsiness_factors.append("head_nodding")
        
        # 2. Check face position (prefer centered faces)
        if current_metrics['center_distance'] > self.POSITION_THRESHOLD:
            drowsiness_score += 0.2
            drowsiness_factors.append("off_center")
        
        # 3. Check face tilt (indicates head dropping)
        if current_metrics['tilt_angle'] > self.HEAD_TILT_THRESHOLD:
            drowsiness_score += 0.3
            drowsiness_factors.append("head_tilt")
        
        # 4. Check face size (too small might indicate leaning back)
        if current_metrics['size_ratio'] < self.FACE_SIZE_THRESHOLD:
            drowsiness_score += 0.3
            drowsiness_factors.append("leaning_back")
        
        # 5. Check vertical position (head dropping)
        if current_metrics['vertical_position'] > 0.6:  # Head is lower in frame
            drowsiness_score += 0.2
            drowsiness_factors.append("head_dropping")
        
        # 6. Check for stability (drowsy people tend to move less)
        if len(self.face_history) > 1:
            last_metrics = self.face_history[-2]
            movement = np.sqrt(
                (current_metrics['center'][0] - last_metrics['center'][0])**2 +
                (current_metrics['center'][1] - last_metrics['center'][1])**2
            )
            if movement < self.MOVEMENT_THRESHOLD:
                drowsiness_score += 0.2
                drowsiness_factors.append("reduced_movement")
        
        # Determine if drowsy based on combined score and factors
        # More lenient threshold but require more factors
        is_drowsy = drowsiness_score >= self.DROWSY_SCORE_THRESHOLD and len(drowsiness_factors) >= 2
        
        # If head nodding is detected, lower the threshold
        if "head_nodding" in drowsiness_factors:
            is_drowsy = drowsiness_score >= (self.DROWSY_SCORE_THRESHOLD * 0.8)
        
        return is_drowsy, drowsiness_score

    def process_drowsiness_frame(self, frame):
        results = self.drowsiness_model(frame, conf=0.25)
        drowsy_detected = False
        drowsiness_score = 0.0
        
        # Find the main person (largest and most centered detection)
        main_person = None
        best_score = -1
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Calculate face center and frame center
                face_center_x = (x1 + x2) / 2
                face_center_y = (y1 + y2) / 2
                
                # Calculate distance from frame center (normalized)
                center_distance = np.sqrt(
                    ((face_center_x - frame_center_x) / frame_width) ** 2 +
                    ((face_center_y - frame_center_y) / frame_height) ** 2
                )
                
                # Calculate face size relative to frame
                face_area = (x2 - x1) * (y2 - y1)
                frame_area = frame_height * frame_width
                relative_size = face_area / frame_area
                
                # Calculate score based on:
                # 1. Confidence (30%)
                # 2. Center position (40%)
                # 3. Relative size (30%)
                position_score = 1 - center_distance  # Closer to center = higher score
                size_score = min(relative_size * 10, 1.0)  # Normalize size score
                score = (conf * 0.3) + (position_score * 0.4) + (size_score * 0.3)
                
                # Only update if this is the best score so far
                if score > best_score:
                    best_score = score
                    main_person = (x1, y1, x2, y2, conf, cls_id)
        
        # Process only the main person if found
        if main_person and best_score > 0.3:  # Only process if we have a good detection
            x1, y1, x2, y2, conf, cls_id = main_person
            
            # Enhanced drowsiness detection
            is_drowsy, drowsiness_score = self.is_drowsy((x1, y1, x2, y2), frame.shape, conf)
            
            if is_drowsy:
                drowsy_detected = True
                self.drowsy_counter += 1
                if self.drowsy_counter >= self.DROWSY_CONSEC_FRAMES:
                    if not self.ALARM_ON:
                        self.ALARM_ON = True
                        self.alert_sound.play()
                    # Draw drowsiness alert with red box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "DROWSY", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "⚠️ DROWSINESS ALERT! ⚠️", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                self.drowsy_counter = max(0, self.drowsy_counter - 1)  # Gradual decrease
                self.ALARM_ON = False
                # Draw awake state with green box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "AWAKE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            self.drowsy_counter = 0
            self.ALARM_ON = False
            # Add message when no main person is detected
            cv2.putText(frame, "No main person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def process_traffic_frame(self, frame):
        results = self.traffic_model(frame)
        frame_detections = {class_name: 0 for class_name in self.traffic_classes.values()}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in self.traffic_classes and conf > 0.5:
                    class_name = self.traffic_classes[cls]
                    # Update detection counts
                    frame_detections[class_name] += 1
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update session state detection counts
        if 'detection_counts' in st.session_state:
            for class_name, count in frame_detections.items():
                st.session_state.detection_counts[class_name] += count
        
        return frame

def main():
    st.set_page_config(page_title="Integrated Traffic & Drowsiness Detection",
                      layout="wide")
    
    # Initialize session state
    if 'system' not in st.session_state:
        st.session_state.system = IntegratedSystem()
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Traffic Detection", "Drowsiness Detection"])
    
    def get_laptop_camera():
        """Get the laptop camera feed"""
        try:
            # Try to find an available camera
            camera_index = try_camera_indices()
            if camera_index is None:
                st.error("Could not find an available camera")
                return None
            
            # Open the camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                st.error("Could not open camera")
                return None
            return cap
        except Exception as e:
            st.error(f"Error accessing camera: {str(e)}")
            return None
    
    if page == "Drowsiness Detection":
        st.title("Live Drowsiness Detection")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Live Camera Feed")
            
            # Start/Stop button
            if st.button("Start/Stop Camera"):
                if not st.session_state.camera_on:
                    # Start camera
                    cap = get_laptop_camera()
                    if cap is not None:
                        st.session_state.camera_on = True
                        st.session_state.cap = cap
                else:
                    # Stop camera
                    if hasattr(st.session_state, 'cap'):
                        st.session_state.cap.release()
                    st.session_state.camera_on = False
            
            # Camera feed placeholder
            camera_placeholder = st.empty()
            
            if st.session_state.camera_on and hasattr(st.session_state, 'cap'):
                try:
                    while st.session_state.camera_on:
                        ret, frame = st.session_state.cap.read()
                        if not ret:
                            st.error("Failed to capture frame from camera")
                            break
                        
                        try:
                            # Process frame for drowsiness detection
                            frame = st.session_state.system.process_drowsiness_frame(frame)
                            
                            # Convert to RGB for display
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Display the frame with container width
                            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        except Exception as e:
                            # If processing fails, show the original frame
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                            print(f"Error processing frame: {str(e)}")
                        
                        # Add a small delay to control frame rate
                        time.sleep(0.01)
                except Exception as e:
                    st.error(f"Error during camera operation: {str(e)}")
                    if hasattr(st.session_state, 'cap'):
                        st.session_state.cap.release()
                    st.session_state.camera_on = False
            else:
                if not st.session_state.camera_on:
                    camera_placeholder.info("Click 'Start/Stop Camera' to begin detection")
                else:
                    camera_placeholder.error("Camera not available")
        
        with col2:
            st.markdown("### Detection Status")
            if st.session_state.camera_on and hasattr(st.session_state, 'cap'):
                st.success("Camera is active")
                st.markdown("""
                ### Detection Guide:
                - 🟢 Green box: Awake
                - 🔴 Red box: Drowsy
                - 🔊 Alert will sound when drowsiness is detected
                
                ### Tips:
                1. Ensure good lighting
                2. Face the camera directly
                3. Keep your face within the frame
                4. Stay still for better detection
                5. The system needs to detect drowsiness for 20 consecutive frames to trigger an alert
                """)
            else:
                st.info("Camera is off")
    
    else:  # Traffic Detection page
        st.title("🚗 Traffic Analysis Dashboard")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Traffic Detection")
            
            # Video source selection
            video_source = st.radio(
                "Select Video Source",
                ["Use Webcam", "Upload Video"],
                horizontal=True
            )
            
            video_placeholder = st.empty()
            
            if video_source == "Use Webcam":
                # Start/Stop button for webcam
                if st.button("Start/Stop Webcam"):
                    if not st.session_state.camera_on:
                        # Start camera
                        cap = get_laptop_camera()
                        if cap is not None:
                            st.session_state.camera_on = True
                            st.session_state.cap = cap
                    else:
                        # Stop camera
                        if hasattr(st.session_state, 'cap'):
                            st.session_state.cap.release()
                        st.session_state.camera_on = False
                
                if st.session_state.camera_on and hasattr(st.session_state, 'cap'):
                    try:
                        while st.session_state.camera_on:
                            ret, frame = st.session_state.cap.read()
                            if not ret:
                                st.error("Failed to capture frame from camera")
                                break
                            
                            # Process frame
                            processed_frame = st.session_state.system.process_traffic_frame(frame)
                            
                            # Convert to RGB for display
                            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(processed_frame_rgb, channels="RGB")
                            
                            # Add small delay
                            time.sleep(0.01)
                    except Exception as e:
                        st.error(f"Error during camera operation: {str(e)}")
                        if hasattr(st.session_state, 'cap'):
                            st.session_state.cap.release()
                        st.session_state.camera_on = False
                else:
                    if not st.session_state.camera_on:
                        video_placeholder.info("Click 'Start/Stop Webcam' to begin detection")
                    else:
                        video_placeholder.error("Camera not available")
            
            else:  # Upload Video mode
                uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
                
                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_file.read())
                    
                    # Get video properties
                    cap = cv2.VideoCapture(tfile.name)
                    if not cap.isOpened():
                        st.error("Failed to open uploaded video file")
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Add progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Process video
                        frame_idx = 0
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Process frame
                            processed_frame = st.session_state.system.process_traffic_frame(frame)
                            
                            # Convert to RGB for display
                            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(processed_frame_rgb, channels="RGB")
                            
                            # Update progress
                            frame_idx += 1
                            progress = frame_idx / frame_count
                            progress_bar.progress(progress)
                            status_text.text(f"Processing frame {frame_idx} of {frame_count}")
                            
                            # Add small delay to control playback speed
                            time.sleep(1/fps)
                        
                        cap.release()
                        os.unlink(tfile.name)
                        
                        st.success("Video processing completed!")

if __name__ == "__main__":
    main() 