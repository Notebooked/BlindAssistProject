import cv2
import numpy as np
import torch
import time
import threading
import subprocess
import sys
import platform
from ultralytics import YOLO

# Load both models
yolov8_model = YOLO('/Users/rosha/GitHub/BlindAssistProject/runs/detect/train/weights/best.pt')
yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# TTS threading variables
tts_lock = threading.Lock()
tts_queue = []
tts_thread_active = False

def speak_using_system(text):
    """Use system TTS instead of pyttsx3"""
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["say", text], check=False)
        elif system == "Windows":
            subprocess.run(["powershell", "-Command", f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"], check=False)
        elif system == "Linux":
            subprocess.run(["espeak", text], check=False)
        else:
            print(f"TTS: {text}")  # Fallback to print
    except Exception as e:
        print(f"TTS Error: {e}")
        print(f"TTS: {text}")  # Fallback to print

def tts_worker():
    """Worker function to handle TTS in a separate thread"""
    global tts_thread_active
    
    print("TTS worker thread started")
    
    while tts_thread_active:
        with tts_lock:
            if tts_queue:
                text = tts_queue.pop(0)
            else:
                text = None
        
        if text:
            try:
                print(f"Speaking: {text}")
                speak_using_system(text)
                print("TTS completed successfully")
            except Exception as e:
                print(f"TTS error: {e}")
        else:
            time.sleep(0.1)  # Short sleep to prevent busy waiting
    
    print("TTS worker thread ending")

def speak_text(text):
    """Add text to TTS queue"""
    with tts_lock:
        tts_queue.clear()  # Clear queue to avoid backlog
        tts_queue.append(text)

print("YOLOv8 Model loaded successfully!")
print(f"YOLOv8 Model classes: {yolov8_model.names}")
print(f"YOLOv8 Number of classes: {len(yolov8_model.names)}")

print("\nYOLOv5 Model loaded successfully!")
print(f"YOLOv5 Model classes: {yolov5_model.names}")
print(f"YOLOv5 Number of classes: {len(yolov5_model.names)}")

confidence_threshold = 0.1
tts_interval = 10  # TTS announcement every 5 seconds
last_tts_time = 0

# Start TTS worker thread
tts_thread_active = True
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame_interval = 1  # frame interval for how often it runs the models

    if frame_count % frame_interval == 0:
        # Run both models on the same frame
        yolov8_results = yolov8_model(frame, verbose=False)
        yolov5_results = yolov5_model(frame)  # PyTorch Hub YOLOv5 inference
        
        # Process YOLOv8 results
        yolov8_detections = 0
        if yolov8_results and len(yolov8_results) > 0:
            result = yolov8_results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                yolov8_detections = len(result.boxes)
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                print(f"Frame {frame_count} - YOLOv8: Found {yolov8_detections} detections")
                print(f"YOLOv8 detected classes: {[yolov8_model.names[int(cls)] for cls in class_ids]}")
        
        # Process YOLOv5 results (different format from PyTorch Hub)
        yolov5_detections = 0
        if yolov5_results is not None:
            # PyTorch Hub YOLOv5 returns a Results object with .pandas() method
            detections = yolov5_results.pandas().xyxy[0]  # pandas DataFrame
            yolov5_detections = len(detections)
            if yolov5_detections > 0:
                print(f"Frame {frame_count} - YOLOv5: Found {yolov5_detections} detections")
                print(f"YOLOv5 detected classes: {detections['name'].tolist()}")

    # Collect YOLOv8 detections first (priority)
    yolov8_detections = []  # Store (box, class_name) tuples
    detected_objects = []  # Store (class_name, confidence) for TTS
    total_detections = 0
    
    if yolov8_results is not None and len(yolov8_results) > 0:
        result = yolov8_results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score < confidence_threshold:
                    continue

                total_detections += 1
                x1, y1, x2, y2 = map(int, box)
                class_name = yolov8_model.names[class_id]
                yolov8_detections.append(((x1, y1, x2, y2), class_name))
                detected_objects.append((class_name, score))  # Add to TTS list

                # Green boxes for YOLOv8
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label_text = f"v8: {class_name} {score:.2f}"

                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                
                cv2.putText(frame, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw YOLOv5 detections only if they're not the same object as YOLOv8 detections
    if yolov5_results is not None:
        # Get detections from PyTorch Hub YOLOv5
        detections = yolov5_results.pandas().xyxy[0]
        
        for _, detection in detections.iterrows():
            score = detection['confidence']
            if score < confidence_threshold:
                continue

            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            current_box = (x1, y1, x2, y2)
            current_class = detection['name']
            
            # Check if this is the same object (same class + overlapping location)
            is_same_object = False
            for yolov8_box, yolov8_class in yolov8_detections:
                # Check if it's the same class AND overlapping location
                if current_class == yolov8_class:
                    iou = calculate_iou(current_box, yolov8_box)
                    if iou > 0.3:  # Same object if same class and overlapping
                        is_same_object = True
                        break
            
            # Only draw and count if it's NOT the same object
            if not is_same_object:
                total_detections += 1
                detected_objects.append((current_class, score))  # Add to TTS list

                # Blue boxes for YOLOv5
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                label_text = f"v5: {current_class} {score:.2f}"

                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (255, 0, 0), -1)
                
                cv2.putText(frame, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display total detection count
    cv2.putText(frame, f"Total Detections: {total_detections}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display legend
    cv2.putText(frame, "Green: YOLOv8 | Blue: YOLOv5", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # TTS announcement every 5 seconds
    current_time = time.time()
    if current_time - last_tts_time >= tts_interval:
        if detected_objects:
            # Group objects by class and find highest confidence for each
            object_dict = {}
            for obj_name, confidence in detected_objects:
                if obj_name not in object_dict or confidence > object_dict[obj_name]:
                    object_dict[obj_name] = confidence
            
            text = "I see the following objects: " + ", ".join(
                [f"{obj} with {conf*100:.0f} percent confidence" for obj, conf in object_dict.items()]
            )
        else:
            text = "I do not see any objects with high confidence."
        
        speak_text(text)  # Use threaded TTS
        last_tts_time = current_time

    cv2.imshow("Dual YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up TTS thread
tts_thread_active = False
print("Shutting down TTS thread...")
if tts_thread.is_alive():
    tts_thread.join(timeout=2.0)
    if tts_thread.is_alive():
        print("TTS thread did not shut down gracefully")
    else:
        print("TTS thread shut down successfully")

cap.release()
cv2.destroyAllWindows()
print("Program ended successfully")