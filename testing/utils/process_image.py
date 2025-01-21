import time
import torch
import os
from PIL import Image
from torchvision import transforms
from utils.metric_calculation import calculate_bpp, calculate_psnr, calculate_lpips

"""
This function processes a single image through the LIVE network, saving versions of the image at various steps, including a reference image decoded from latent space without further compression.
It then calls a function to save jpeg, webp, and heif compressed versions of the latent image.
Finally, it calculates and stores the bpp, psnr, and lpips metrics for the compressed formats and the image that goes through LIVE compression.
Args:
    --batch_tensor: The tensor of the image to be processed.
    --image: The original image.
    --image_name: The name of the image to be processed.
    --net: The defined LIVE model.
    --ae: The autoencoder.
    --output_path: The path where the all output images are to be saved.
    --metrics: The dictionary to store all the metrics.
    --to_print: Whether or not to print the metrics.
"""
def process_image(batch_tensor: torch.Tensor, image: Image, image_name: str, net, ae, output_path: str, metrics: dict[str, dict[str, list]], to_print: bool = False):

    with torch.no_grad():
        # Encodes image to latent space
        img_latent = generate_latent_image(batch_tensor, os.path.join(output_path, f'{image_name}_ae_decomp.pt'), ae, save_encode_time=False)
        
        # Decodes from latent space directly, without going through net
        precomp_img = generate_precomp_image(img_latent, os.path.join(output_path, f'{image_name}_precomp_img.png'), ae, net)
        
        # Saves compressed image during LIVE compression
        y_strings, compressed_output = generate_y_strings(img_latent, os.path.join(output_path, f'{image_name}_y_strings.pt'), net, save_y_strings_time=False)

        # Process final output (image decompressed from net then decoded from latent space)
        compressed_image = generate_compressed_image(y_strings, compressed_output, os.path.join(output_path, f'{image_name}_compressed.png'), net, ae, save_decompress_time=False)
        
        # Save latent image in PNG format (no special parameters needed) - will use as reference image for later
        ae_decompressed_image = decompress_latent(img_latent, os.path.join(output_path, f'{image_name}_reference_ae.png'), ae, save_decode_time=False)

        for key in metrics.keys():
            if 'compressed' in key:
                quality = key.split('_qual_')[1]
                file_name = key.split('_qual_')[0]
                save_ref_image(quality, file_name, ae_decompressed_image, output_path, image_name)
        
        return calculate_metrics(metrics, image, image_name, output_path, compressed_image, precomp_img, to_print)


"""These helper functions below are called by process_image to generate images at different steps of the pipeline,
save reference images for specified compression formats, and calculate metrics for all formats"""
"""NOTE: These functions can also be called individually (without process_image) in the sweep_all.ipynb file"""
def generate_latent_image(batch_tensor: torch.Tensor, output_path: str, ae, save_encode_time: bool = False):
    encode_start_time = time.time()
    img_transform = ae.encode(batch_tensor).latent_dist
    img_latent = img_transform.sample()
    encode_time = time.time() - encode_start_time
    torch.save(img_latent, output_path)
    if save_encode_time:
        return img_latent, encode_time
    else:
        return img_latent
    
def decompress_latent(img_latent: torch.Tensor, output_path: str, ae, save_decode_time: bool = False):
    decode_start_time = time.time()
    ae_decompressed = ae.decode(img_latent)
    decode_time = time.time() - decode_start_time
    ae_decompressed_image = transforms.ToPILImage()(ae_decompressed.sample.squeeze(0).clamp(0, 1))
    ae_decompressed_image.save(output_path)
    if save_decode_time:
        return ae_decompressed_image, decode_time
    else:
        return ae_decompressed_image

def generate_precomp_image(latent_image: torch.Tensor, output_path: str, ae, net):
    precomp = ae.decode(net(latent_image)['x_hat']).sample
    precomp_img = transforms.ToPILImage()(precomp.squeeze(0).clamp(0, 1))
    precomp_img.save(output_path)
    return precomp_img

def generate_y_strings(img_latent: torch.Tensor, output_path: str, net, save_y_strings_time: bool = False):
    start_time = time.time()
    compressed_output = net.compress(img_latent)
    y_strings_time = time.time() - start_time
    y_strings = compressed_output['strings']
    torch.save(y_strings, output_path)
    if save_y_strings_time:
        return y_strings, compressed_output,y_strings_time
    else:
        return y_strings, compressed_output

def generate_compressed_image(y_strings: torch.Tensor, compressed_output: dict, output_path: str, net, ae, save_decompress_time: bool = False):
    start_time = time.time()
    output = net.decompress(y_strings, compressed_output['shape'])
    decompress_time = time.time() - start_time
    output_img = ae.decode(output['x_hat']).sample
    compressed_image = transforms.ToPILImage()(output_img.squeeze(0).clamp(0, 1))
    compressed_image.save(output_path)
    if save_decompress_time:
        return compressed_image, decompress_time
    else:
        return compressed_image

"""Called only by sweep_all.ipynb to determine time needed for LIVE compression without arithmetic encoding""" 
def get_no_strings_time(latent_image: torch.Tensor, net):
    start_time = time.time()
    net(latent_image)['x_hat']
    no_strings_time = time.time() - start_time
    return no_strings_time


"""Helper function to save reference images for specified compression formats

Args:
    --quality: compression quality.
    --file_name: The name of the file to be saved.
    --ref_image: The reference image to be saved.
    --output_path: The path where the reference image will be saved.
    --image_name: The name of the image.
"""
def save_ref_image(quality, file_name, ref_image: Image, output_path: str, image_name: str):
    output_path_format = os.path.join(output_path, f'{image_name}_{file_name}')
    ref_image.save(output_path_format, quality=int(quality))


"""Helper function to calculate metrics for all compression formats
Args:
    --metrics: The dictionary to store all the metrics.
    --image: The original image.
    --image_name: The name of the image.
    --output_path: The path where the all output images are saved.
    --compressed_image: The compressed image.
    --precomp_img: The precomputed image.
    --to_print: Whether or not to print the metrics.
"""
def calculate_metrics(metrics: dict[str, dict[str, list]], image: Image, image_name: str, output_path: str, compressed_image: Image, precomp_img: Image, to_print: bool):
    num_pixels = image.size[0] * image.size[1]
    reference_ae = Image.open(os.path.join(output_path, f'{image_name}_reference_ae.png'))

    for key in metrics.keys():
        # Get the image to process based on the key
        current_image = None
        match key:
            case 'live_hp':
                current_image = compressed_image
                file_path = os.path.join(output_path, f'{image_name}_y_strings.pt')
            case 'live_hp_precomp':
                current_image = precomp_img
                file_path = os.path.join(output_path, f'{image_name}_y_strings.pt')
            case 'reference_ae':
                current_image = reference_ae
                file_path = os.path.join(output_path, f'{image_name}_{key}.png')
            case _ if 'compressed' in key:
                file_name = key.split('_qual_')[0]
                try:
                    current_image = Image.open(os.path.join(output_path, f'{image_name}_{file_name}'))
                    file_path = os.path.join(output_path, f'{image_name}_{file_name}')
                except Exception as e:
                    print(f"Warning: Could not load image for {key}: {e}")
            # If the key is not a valid compression format, skip metric calculation for it
            case _:
                continue

        if current_image is not None:
            # Calculate metrics
            results = {
                'bpp': calculate_bpp(file_path, num_pixels),
                'psnr': calculate_psnr(reference_ae, current_image),
                'lpips': calculate_lpips(reference_ae, current_image)
            }
            
            # Store metrics
            for metric_name, metric_value in results.items():
                if metric_name in metrics[key]:
                    metrics[key][metric_name].append(metric_value)
                    if to_print:
                        print(f'{metric_name.upper()} for {image_name} ({key}): {metric_value:.4f}')
    
    return metrics