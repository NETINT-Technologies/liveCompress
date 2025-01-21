"""Predefined algorithm for compressing an image with JPEG.
Args:
    --input_dir: The path to the directory containing the images.
    --output_image_path: The path where the output image file will be saved.
"""
from PIL import Image
import os

# Define the directory containing the images
input_dir = '/home/ai1/ae-compress-1/stage_1_simple_ae/train/sequences/train'
output_image_path = '/home/ai1/ae-compress-1/manual_testing/jpg_compressed_image.jpg'

# Get the first image in the directory
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
if not image_files:
    raise FileNotFoundError("No image files found in the directory.")

first_image_path = os.path.join(input_dir, image_files[12])

# Open the first image
image = Image.open(first_image_path)

# Save the image with JPEG compression
image.save(output_image_path, 'JPEG', quality=10)  # You can adjust the quality parameter as needed

print(f"First image compressed and saved at {output_image_path}")
