"""This script sweeps through all checkpoints of live hp and calculates psnr, lpips, and bpp for all images in the kodak dataset
It also compares them with the psnr, lpips, and bpp of the images compressed with jpeg, webp, and heif
Comprehensive results for all 24 images compressed with all 4 methods are saved in an excel file, with results for each metric seperately saved in individual csv files.
NOTE: Unlike the ipynb file, this script does not go through different quality settings for jpeg, webp, and heif."""

import torch
from torchvision import transforms
import os
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
from pillow_heif import register_heif_opener
import pandas as pd

from utils.preprocess_image import preprocess_image
from utils.load_checkpoints import load_checkpoints
from utils.process_image import process_image
import sys
sys.path.append('..')
from model.model import liveModel

register_heif_opener()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to process images and calculate metrics
def process_images(checkpoint_path):
    net = liveModel(N=24, M=36).to(device)
    load_checkpoints(checkpoint_path, net, device)
    print(f'Processing checkpoint: {checkpoint_path}')

    #create metrics, however only include live_hp_precomp and compressed latent images
    metrics = {
        'live_hp_precomp': {'bpp': [], 'psnr': [], 'lpips': []},
        'compressed_jpg.jpeg_qual_25': {'bpp': [], 'psnr': [], 'lpips': []},
        'compressed_webp.webp_qual_42': {'bpp': [], 'psnr': [], 'lpips': []},
        'compressed_heif.heif_qual_40': {'bpp': [], 'psnr': [], 'lpips': []}
    }

    for image_name_bmp in os.listdir(images_path):
        image, image_name = preprocess_image(images_path, image_name_bmp, output_path)
        if image is None or image_name is None:
            continue
        # Convert image to tensor
        tensors = transforms.ToTensor()(image).unsqueeze(0)
        batch_tensor = tensors.to(device)
        net.update()

        metrics = process_image(batch_tensor, image, image_name, net, ae, output_path, metrics, False)
    return metrics

def generate_csvs(metrics_by_checkpoint: dict[str, dict[str, dict[str, list]]]):
    # First, create a list of column names (one for each image in the Kodak dataset)
    column_names = [img_name for img_name in os.listdir(images_path)]
    column_names.remove('readme.md')

    # Initialize empty DataFrames with the correct columns
    bpp_df = pd.DataFrame(columns=column_names)
    psnr_df = pd.DataFrame(columns=column_names)
    lpips_df = pd.DataFrame(columns=column_names)

    # Populate the DataFrames
    for checkpoint, metrics in metrics_by_checkpoint.items():
        for key, metric_values in metrics.items():
            bpp_df.loc[f'{checkpoint}_{key}'] = metric_values['bpp']
            psnr_df.loc[f'{checkpoint}_{key}'] = metric_values['psnr']
            lpips_df.loc[f'{checkpoint}_{key}'] = metric_values['lpips']

    # Add average column to each DataFrame
    bpp_df['Average'] = bpp_df.mean(axis=1)
    psnr_df['Average'] = psnr_df.mean(axis=1)
    lpips_df['Average'] = lpips_df.mean(axis=1)

    # Save each DataFrame to a separate sheet in an excel file
    with pd.ExcelWriter('all_metrics_by_checkpoint_data.xlsx') as writer:
        bpp_df.to_excel(writer, sheet_name='bpp')
        psnr_df.to_excel(writer, sheet_name='psnr')
        lpips_df.to_excel(writer, sheet_name='lpips')

    # Save each dataframe to an individual csv file as backup
    bpp_df.to_csv('bpp_by_checkpoint_data.csv')
    psnr_df.to_csv('psnr_by_checkpoint_data.csv')
    lpips_df.to_csv('lpips_by_checkpoint_data.csv')

# Initialize autoencoder
ae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

# Paths
images_path = '../kodak'
output_path = './outputs'
checkpoints_dir = '../checkpoints'

# Collect metrics for all checkpoints
all_metrics = {
    'live_hp_precomp': {'bpp': [], 'psnr': [], 'lpips': []},
    'compressed_jpg.jpeg_qual_25': {'bpp': [], 'psnr': [], 'lpips': []},
    'compressed_webp.webp_qual_42': {'bpp': [], 'psnr': [], 'lpips': []},
    'compressed_heif.heif_qual_40': {'bpp': [], 'psnr': [], 'lpips': []}
}

metrics_by_checkpoint = {}

# Process each image through the model, getting the metrics for each method
for checkpoint in os.listdir(checkpoints_dir):
    if checkpoint.endswith('best_loss.pth.tar'):
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
        metrics = process_images(checkpoint_path)

        for key in all_metrics:
            all_metrics[key]['bpp'].extend(metrics[key]['bpp'])
            all_metrics[key]['psnr'].extend(metrics[key]['psnr'])
            all_metrics[key]['lpips'].extend(metrics[key]['lpips'])
        metrics_by_checkpoint[checkpoint] = metrics

# Uses the calculated metrics to create csvs
generate_csvs(metrics_by_checkpoint)