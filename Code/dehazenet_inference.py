# dehazenet_inference.py

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from model import DehazeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple ToTensor transform
transform = T.Compose([T.ToTensor()])

def dehaze_video(input_path: str, output_path: str):
    # Initialize model
    model = DehazeNet().to(device)

    # Load checkpoint and strip 'module.' prefixes if present
    ckpt = torch.load("dehazenet.pth", map_location=device)
    # If checkpoint was saved from DataParallel, keys start with 'module.'
    new_ckpt = {}
    for k, v in ckpt.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_ckpt[new_key] = v
    model.load_state_dict(new_ckpt)
    model.eval()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = transform(img).unsqueeze(0).to(device)

            # Forward pass
            output_tensor = model(tensor).squeeze(0).cpu()

            # Convert back to image
            output_img = (output_tensor * 255).clamp(0,255).permute(1,2,0).numpy().astype(np.uint8)
            bgr_out = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            out.write(bgr_out)

    cap.release()
    out.release()
