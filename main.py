import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('/Users/rosha/GitHub/BlindAssistProject/runs/detect/train/weights/best.pt')

print("Model loaded successfully!")
print(f"Model classes: {model.names}")
print(f"Number of classes: {len(model.names)}")

confidence_threshold = 0.1

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

frame_count = 0
results = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame_interval = 1  # frame interval for how often it runs the model

    if frame_count % frame_interval == 0:
        results = model(frame, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"Frame {frame_count}: Found {len(result.boxes)} detections")
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                print(f"Max confidence: {scores.max():.3f}, Min confidence: {scores.min():.3f}")
                print(f"Detected classes: {[model.names[int(cls)] for cls in class_ids]}")
            else:
                print(f"Frame {frame_count}: No detections")

    if results is not None and len(results) > 0:
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            class_names = model.names
            
            detection_count = 0
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score < confidence_threshold:
                    continue

                detection_count += 1
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label_text = f"{class_names[class_id]} {score:.2f}"

                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                
                cv2.putText(frame, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            cv2.putText(frame, f"Detections: {detection_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()