import torch
from PIL import Image
import torchvision.transforms as transforms
from model import GridDehazeNet
import cv2
import os
import numpy as np

# Load model once to reuse across frames
model = GridDehazeNet()
model.load_state_dict(torch.load('griddehaze.pth', map_location='cpu'))
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
])

# Define postprocessing
postprocess = transforms.ToPILImage()

def dehaze_image_cv2_frame(frame):
    """Dehaze a single OpenCV BGR frame using GridDehazeNet."""
    # Convert BGR (OpenCV) to RGB (PIL)
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    # Preprocess and run inference
    input_tensor = preprocess(pil_img).unsqueeze(0)  # [1, 3, H, W]
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Convert output back to image
    output_img = postprocess(output_tensor.squeeze(0))

    # Resize back to original frame size and convert to BGR
    output_img = output_img.resize((frame.shape[1], frame.shape[0]))
    output_bgr = cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR)

    return output_bgr

def grid_dehaze_video(input_path, output_path):
    """Read video, dehaze frame-by-frame, and write output."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Error opening video file.")

    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Output codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dehazed_frame = dehaze_image_cv2_frame(frame)
        out.write(dehazed_frame)

    cap.release()
    out.release()
