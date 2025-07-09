import cv2
import torch
import time
import threading
import subprocess
import platform
from ultralytics import YOLO
import init_da

# Load models
print("Loading models...")
yolov8_model = YOLO('runs/detect/train/weights/best.pt')
yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
init_da.load_model()
print("Models loaded.")

# Shared variables
tts_lock = threading.Lock()
tts_queue = []
tts_thread_active = False

detection_lock = threading.Lock()
latest_detections = {
    "objects": [],
    "boxes": [],
    "frame": None,
}

DEPTH_SCALE = 10
confidence_threshold = 0.1
tts_interval = 10
last_tts_time = 0

def speak_using_system(text):
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["say", text], check=False)
        elif system == "Windows":
            subprocess.run(["powershell", "-Command", f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"], check=False)
        elif system == "Linux":
            subprocess.run(["espeak", text], check=False)
        else:
            print(f"TTS: {text}")
    except Exception as e:
        print(f"TTS Error: {e}")
        print(f"TTS: {text}")

def speak_text(text):
    with tts_lock:
        tts_queue.clear()
        tts_queue.append(text)

def get_position_description(x1, x2, frame_width):
    center_x = (x1 + x2) / 2
    left_threshold = frame_width * 0.35
    right_threshold = frame_width * 0.65
    if center_x < left_threshold:
        return "on the left"
    elif center_x > right_threshold:
        return "on the right"
    else:
        return "in the center"

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def tts_worker():
    global tts_thread_active, latest_detections

    print("TTS worker thread started")

    while tts_thread_active:
        with tts_lock:
            if tts_queue:
                text = tts_queue.pop(0)
            else:
                text = None

        if text:
            speak_using_system(text)
            print("TTS completed, now running inference...")

            ret, frame = cap.read()
            if not ret:
                continue

            frame_height, frame_width = frame.shape[:2]
            yolov8_results = yolov8_model(frame, verbose=False)
            yolov5_results = yolov5_model(frame)

            yolov8_detections = []
            detected_objects = []

            if yolov8_results:
                result = yolov8_results[0]
                if result.boxes:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for box, score, class_id in zip(boxes, scores, class_ids):
                        if score < confidence_threshold:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        class_name = yolov8_model.names[class_id]
                        position = get_position_description(x1, x2, frame_width)
                        depth = init_da.get_avg_depth_in_image(frame, x1, x2, y1, y2) * DEPTH_SCALE
                        detected_objects.append((class_name, score, position, depth))
                        yolov8_detections.append(((x1, y1, x2, y2), f"v8: {class_name} {score:.2f}", (0, 255, 0)))

            if yolov5_results is not None:
                detections = yolov5_results.pandas().xyxy[0]
                for _, detection in detections.iterrows():
                    score = detection['confidence']
                    if score < confidence_threshold:
                        continue
                    x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                    current_box = (x1, y1, x2, y2)
                    current_class = detection['name']
                    duplicate = False
                    for v8_box, _, _ in yolov8_detections:
                        if calculate_iou(current_box, v8_box) > 0.3:
                            duplicate = True
                            break
                    if not duplicate:
                        position = get_position_description(x1, x2, frame_width)
                        depth = init_da.get_avg_depth_in_image(frame, x1, x2, y1, y2) * DEPTH_SCALE
                        detected_objects.append((current_class, score, position, depth))
                        yolov8_detections.append(((x1, y1, x2, y2), f"v5: {current_class} {score:.2f}", (255, 0, 0)))

            with detection_lock:
                latest_detections["objects"] = detected_objects
                latest_detections["boxes"] = yolov8_detections
                latest_detections["frame"] = frame.copy()

        else:
            time.sleep(0.1)

    print("TTS worker thread ending")

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

# Start TTS thread
tts_thread_active = True
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Main loop
while True:
    ret, live_frame = cap.read()
    if not ret:
        continue

    with detection_lock:
        boxes = latest_detections["boxes"]
        detected_objects = latest_detections["objects"]

    for (x1, y1, x2, y2), label, color in boxes:
        cv2.rectangle(live_frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(live_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(live_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    left_line = int(live_frame.shape[1] * 0.35)
    right_line = int(live_frame.shape[1] * 0.65)
    cv2.line(live_frame, (left_line, 0), (left_line, live_frame.shape[0]), (128, 128, 128), 1)
    cv2.line(live_frame, (right_line, 0), (right_line, live_frame.shape[0]), (128, 128, 128), 1)

    cv2.putText(live_frame, f"Total Detections: {len(detected_objects)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(live_frame, "Green: YOLOv8 | Blue: YOLOv5", (10, live_frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(live_frame, "Gray lines: Left/Center/Right zones", (10, live_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Dual YOLO Object Detection", live_frame)

    current_time = time.time()
    if current_time - last_tts_time >= tts_interval:
        if detected_objects:
            summary = []
            grouped = {}
            for name, conf, pos, depth in detected_objects:
                key = f"{name}_{pos}"
                if key not in grouped or conf > grouped[key][1]:
                    grouped[key] = (name, conf, pos, depth)
            for name, conf, pos, depth in grouped.values():
                summary.append(f"{name} {pos}, approximately {depth:.2f} meters away, with {conf*100:.0f} percent confidence")
            speak_text("I see the following objects: " + ", ".join(summary))
        else:
            speak_text("I do not see any objects with high confidence.")
        last_tts_time = current_time

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
tts_thread_active = False
if tts_thread.is_alive():
    tts_thread.join(timeout=2.0)
cap.release()
cv2.destroyAllWindows()
print("Program ended successfully")