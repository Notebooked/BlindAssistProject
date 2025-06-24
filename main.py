import cv2
import torch
from ultralytics import YOLO

yolov8_model = YOLO('/Users/rosha/GitHub/BlindAssistProject/runs/detect/train/weights/best.pt')
yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

print("YOLOv8 Model loaded successfully!")
print(f"YOLOv8 Model classes: {yolov8_model.names}")
print(f"YOLOv8 Number of classes: {len(yolov8_model.names)}")

print("\nYOLOv5 Model loaded successfully!")
print(f"YOLOv5 Model classes: {yolov5_model.names}")
print(f"YOLOv5 Number of classes: {len(yolov5_model.names)}")

confidence_threshold = 0.1

# iou is intersection over union, checking for yolo8 and 5 perhaps double counting

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

    frame_interval = 1  # interval for how often models are fun, bigger number means less frequent checks

    if frame_count % frame_interval == 0:
        yolov8_results = yolov8_model(frame, verbose=False)
        yolov5_results = yolov5_model(frame)  
        
        yolov8_detections = 0
        if yolov8_results and len(yolov8_results) > 0:
            result = yolov8_results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                yolov8_detections = len(result.boxes)
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                print(f"Frame {frame_count} - YOLOv8: Found {yolov8_detections} detections")
                print(f"YOLOv8 detected classes: {[yolov8_model.names[int(cls)] for cls in class_ids]}")
        
        yolov5_detections = 0
        if yolov5_results is not None:
            detections = yolov5_results.pandas().xyxy[0]  
            yolov5_detections = len(detections)
            if yolov5_detections > 0:
                print(f"Frame {frame_count} - YOLOv5: Found {yolov5_detections} detections")
                print(f"YOLOv5 detected classes: {detections['name'].tolist()}")

    yolov8_detections = []  
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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label_text = f"v8: {class_name} {score:.2f}"

                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                
                cv2.putText(frame, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if yolov5_results is not None:
        detections = yolov5_results.pandas().xyxy[0]
        
        for _, detection in detections.iterrows():
            score = detection['confidence']
            if score < confidence_threshold:
                continue

            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            current_box = (x1, y1, x2, y2)
            current_class = detection['name']
            
            is_same_object = False
            for yolov8_box, yolov8_class in yolov8_detections:
                if current_class == yolov8_class:
                    iou = calculate_iou(current_box, yolov8_box)
                    if iou > 0.3:  
                        is_same_object = True
                        break
            
            if not is_same_object:
                total_detections += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                label_text = f"v5: {current_class} {score:.2f}"

                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (255, 0, 0), -1)
                
                cv2.putText(frame, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Total Detections: {total_detections}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.putText(frame, "Green: YOLOv8 | Blue: YOLOv5", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Dual YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
