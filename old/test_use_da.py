import cv2
import torch
import os
import requests

DA_PATH = 'depth_anything_v2_vits.pth'
DA_URL = 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true'
if not os.path.exists(DA_PATH):
    print("Downloading model...")
    response = requests.get(DA_URL, allow_redirects=True)
    if response.status_code == 200:
        with open(DA_PATH, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        raise Exception(f"Failed to download model. Status code: {response.status_code}")
else:
    print("Model already exists.")

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

model = DepthAnythingV2(**model_configs['vits'])
model.load_state_dict(torch.load(DA_PATH, map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('test_img.png')

import time

start = time.time()
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
print(time.time() - start)

print(depth)