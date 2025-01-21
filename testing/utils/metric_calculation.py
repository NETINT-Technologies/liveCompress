from torchvision import transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
psnr_model = PeakSignalNoiseRatio().to(device)
lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

"""A helper function to calculate PSNR.

Args:
    --original: The original, uncompressed image.
    --compressed: The compressed image.
"""
def calculate_psnr(original, compressed):
    original_tensor = transforms.ToTensor()(original).unsqueeze(0).to(device)
    compressed_tensor = transforms.ToTensor()(compressed).unsqueeze(0).to(device)
    return psnr_model(original_tensor, compressed_tensor).item()

"""A helper function to calculate LPIPS.

Args:
    --original: The original, uncompressed image.
    --compressed: The compressed image.
"""
def calculate_lpips(original, compressed):
    original_tensor = transforms.ToTensor()(original).unsqueeze(0).to(device)
    compressed_tensor = transforms.ToTensor()(compressed).unsqueeze(0).to(device)
    return lpips_model(original_tensor, compressed_tensor).item()

"""A helper function to calculate bits per pixel (bpp).

Args:
    --file_path: The path to the image file to be evaluated.
    --num_pixels: The number of pixels in the original, uncompressed image.
"""
def calculate_bpp(file_path, num_pixels):
    file_size_bits = os.path.getsize(file_path) * 8
    bpp = file_size_bits / num_pixels
    return bpp