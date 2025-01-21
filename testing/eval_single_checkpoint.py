import torch
from torchvision import transforms
import os
from PIL import Image
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
from pillow_heif import register_heif_opener
from utils.preprocess_image import preprocess_image
from utils.process_image import process_image
from utils.load_checkpoints import load_checkpoints
import sys
sys.path.append('..')
from model import newModel
import argparse

register_heif_opener()

"""Helper function to calculate the average of a metric for a given set of values
Args:
    --metric: The metric to calculate the average of (e.g. 'bpp', 'psnr', 'lpips').
    --values: The dictionary of values to calculate the average of.
"""
def calculate_average_metrics(metric, values):
    if metric in values:
        average_metric = sum(values[metric]) / len(values[metric]) if values[metric] else 0
        print(f'Average {metric} for {file_type}: {average_metric:.4f}')


"""Script used for manually testing and plotting metrics of all images for a single checkpoint of a trained model.
Args:
    --checkpoint: The path to the checkpoint file within the checkpoints folder.
"""
def plot_metrics(metrics):
    file_types = list(metrics.keys())
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    markers = ['o', 's', 'D', 'v', 'p', '*']

    # Plot bpp vs lpips
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for file_type, color, marker in zip(file_types, colors, markers):
        bpp = metrics[file_type]['bpp']
        lpips = metrics[file_type]['lpips']
        plt.scatter(bpp, lpips, color=color, marker=marker, label=file_type)
    plt.xlabel('Bits per pixel (bpp)')
    plt.ylabel('LPIPS')
    plt.xlim(0, 1)
    plt.ylim(0, 0.5)
    plt.title('bpp vs LPIPS')
    plt.legend()

    # Plot bpp vs psnr
    plt.subplot(1, 2, 2)
    for file_type, color, marker in zip(file_types, colors, markers):
        bpp = metrics[file_type]['bpp']
        psnr = metrics[file_type]['psnr']
        plt.scatter(bpp, psnr, color=color, marker=marker, label=file_type)
    plt.xlabel('Bits per pixel (bpp)')
    plt.ylabel('PSNR')
    plt.xlim(0, 1)
    plt.ylim(20, 40)
    plt.title('bpp vs PSNR')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # Save the plot to a file
    plt.savefig(os.path.join(output_path, 'metrics_plot.png'))

# Gets the checkpoint path from the command line
parser = argparse.ArgumentParser(description='Process model checkpoint path.')
parser.add_argument('--checkpoint', type=str, required=True, help='Name of the checkpoint file')

args = parser.parse_args()

CHECKPOINT_PATH = os.path.join('../checkpoints', args.checkpoint)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model weights
MODEL_PATH = os.path.expanduser(CHECKPOINT_PATH)
ae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
ae = ae.to(device)


net = newModel(N=24, M=36).eval().to(device) 
load_checkpoints(MODEL_PATH, net, device)

images_path = os.path.expanduser('../kodak')
output_path = os.path.expanduser('./outputs/eval_single_checkpoint')

# Initialize metrics dictionary for the methods we want to compare and the metrics we want to track
# dictionary format must be standardized. keys are filename_qual_quality, values are lists of metrics
metrics = {
    'reference_ae': {'bpp': [], 'psnr': [], 'lpips': []},
    'live_hp': {'bpp': [], 'psnr': [], 'lpips': []},
    'live_hp_precomp': {'bpp': [], 'psnr': [], 'lpips': []},
    'compressed_jpg.jpeg_qual_10': {'bpp': [], 'psnr': [], 'lpips': []},
    'compressed_webp.webp_qual_17': {'bpp': [], 'psnr': [], 'lpips': []},
    'compressed_heif.heif_qual_28': {'bpp': [], 'psnr': [], 'lpips': []}
}

# Process each image in the directory
for image_name_bmp in os.listdir(images_path):
    image, image_name = preprocess_image(images_path, image_name_bmp, output_path)
    if image is None or image_name is None:
        continue
    tensors = transforms.ToTensor()(image).unsqueeze(0)
    batch_tensor = tensors.to(device)
    net.update()
    
    metrics = process_image(batch_tensor, image, image_name, net, ae, output_path, metrics, to_print=True)

# Calculate and print the average bpp, PSNR, and LPIPS for each file type
for file_type, values in metrics.items():
    calculate_average_metrics('bpp', values)
    calculate_average_metrics('psnr', values)
    calculate_average_metrics('lpips', values)

# Call the function to plot the metrics
plot_metrics(metrics)

