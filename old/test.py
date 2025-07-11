from ultralytics import YOLO

yolov8_model = YOLO('best.pt')

results = yolov8_model('test_img.png')
print(results[0].to_json())