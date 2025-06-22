import torch

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)