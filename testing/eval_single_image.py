"""Script used for manually testing a trained model on a single image.

Args:
    --checkpoint: The path to the checkpoint file within the checkpoints folder.
    --image: The path to the image file within the kodak folder.
"""

import torch
from torchvision import transforms
import os

from PIL import Image

from diffusers.models import AutoencoderKL

from utils.metric_calculation import calculate_bpp
from utils.load_checkpoints import load_checkpoints
import sys
sys.path.append('..')
from model import newModel
from utils.preprocess_image import preprocess_image
from utils.process_image import process_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import argparse

parser = argparse.ArgumentParser(description='Process checkpoint and image file paths.')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument('--image', type=str, required=True, help='Path to the image file. Must be in "kodak" directory')

args = parser.parse_args()

CHECKPOINT = os.path.join('../checkpoints', args.checkpoint)
IMAGE_FILE = args.image

ae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
ae = ae.to(device)

net = newModel(N=24, M=36).eval().to(device)
load_checkpoints(CHECKPOINT, net, device)

images_path = '../kodak'

# Replace the manual image processing with preprocess_image utility
image, image_name = preprocess_image(images_path, IMAGE_FILE, './outputs/eval_single_image')
if image is None:
    print(f"Failed to process image: {IMAGE_FILE}")
    sys.exit(1)

# Convert image to tensor
batch_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

# Initialize metrics dictionary for the methods we want to compare and the metrics we want to track
# dictionary format must be standardized. keys are filename_qual_quality, values are lists of metrics
metrics = {
    'live_hp': {'bpp': []},
    'reference_ae': {'bpp': []},
    'compressed_jpg.jpeg_qual_25': {'bpp': []},
    'compressed_webp_15.webp_qual_15': {'bpp': []},
    'compressed_webp_25.webp_qual_25': {'bpp': []},
    'compressed_webp_20.webp_qual_20': {'bpp': []},
    'compressed_webp_75.webp_qual_75': {'bpp': []}
}

with torch.no_grad():
    net.update()
    # Use process_image utility instead of manual processing
    metrics = process_image(
        batch_tensor=batch_tensor,
        image=image,
        image_name='output',
        net=net,
        ae=ae,
        output_path='./outputs/eval_single_image',
        metrics=metrics,
        to_print=True,
    )
    