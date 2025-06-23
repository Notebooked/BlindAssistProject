from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import shutil
import os

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    print("YOLOv11 model loaded.")

    merged_path = Path('./merged_dataset')

    print(merged_path)
    model.train(data=str(merged_path / 'data.yaml'), epochs=50, imgsz=640, amp=False)

# Paths
# car_path = Path('./car')
# anomaly_path = Path('./road_anomaly')

# merged_path.mkdir(exist_ok=True)

# Read both datasets' class names
# with open(car_path / 'data.yaml', 'r') as f:
#     car_yaml = yaml.safe_load(f)
#     car_classes = car_yaml['names']

# with open(anomaly_path / 'data.yaml', 'r') as f:
#     anomaly_yaml = yaml.safe_load(f)
#     anomaly_classes = anomaly_yaml['names']

# Combine class names and generate a mapping for road_anomaly class IDs
# merged_classes = car_classes + anomaly_classes
# anomaly_class_offset = len(car_classes)

# print("Merged class count:", len(merged_classes))

# Create new data.yaml
# new_yaml = {
#     'train': str(merged_path / 'train/images'),
#     'val': str(merged_path / 'valid/images'),
#     'test': str(merged_path / 'test/images'),
#     'nc': len(merged_classes),
#     'names': merged_classes
# }

# with open(merged_path / 'data.yaml', 'w') as f:
#     yaml.dump(new_yaml, f)

# Copy and merge folders
# def copy_data(src_root, dst_root, class_offset=0, relabel=False):
#     for split in ['train', 'valid', 'test']:
#         for subdir in ['images', 'labels']:
#             src = src_root / split / subdir
#             dst = dst_root / split / subdir
#             dst.mkdir(parents=True, exist_ok=True)
            
#             for file in src.glob('*.jpg' if subdir == 'images' else '*.txt'):
#                 dst_file = dst / file.name

#                 if subdir == 'images' or not relabel:
#                     shutil.copy(file, dst_file)
#                 else:
#                     # Relabel file: adjust class IDs
#                     with open(file, 'r') as rf, open(dst_file, 'w') as wf:
#                         for line in rf:
#                             parts = line.strip().split()
#                             if len(parts) >= 1:
#                                 class_id = int(parts[0]) + class_offset
#                                 new_line = ' '.join([str(class_id)] + parts[1:]) + '\n'
#                                 wf.write(new_line)

# Merge datasets
#copy_data(car_path, merged_path, relabel=False)
#copy_data(anomaly_path, merged_path, class_offset=anomaly_class_offset, relabel=True)

#print("Dataset merged successfully. Starting training...")

# Train YOLOv8 on the merged dataset