from PIL import Image
import os

"""Preprocess an image for the model, by renaming the file and resizing it to 768x512. Supports PNG, BMP, JPG, JPEG, WEBP, HEIC, JFIF images.

Args:
    --images_path: The path to the directory containing the images.
    --image_name: The name of the image to be processed.
    --output_path: The path where the processed image will be saved.
"""
def preprocess_image(images_path, image_name, output_path):
    if image_name.endswith('.png') or image_name.endswith('.bmp') or image_name.endswith('.jpg') or image_name.endswith('.jpeg') or image_name.endswith('.webp') or image_name.endswith('.heic') or image_name.endswith('.jfif'): 
        print(f'Processing image: {image_name}')
        #save original image name for opening the image later
        base_name = image_name

        image_name = image_name.replace('.png', '')
        image_name = image_name.replace('.bmp', '') 
        image_name = image_name.replace('.jpg', '')
        image_name = image_name.replace('.jpeg', '')
        image_name = image_name.replace('.webp', '')
        image_name = image_name.replace('.heic', '')
        image_name = image_name.replace('.jfif', '')

        image = Image.open(os.path.join(images_path, base_name)).convert('RGB')

        # Check orientation and resize accordingly
        if image.width > image.height:
            image = image.resize((768, 512))
        else:
            image = image.resize((512, 768))
        
        # Save the reference image
        reference_image_path = os.path.join(output_path, f'{image_name}_reference.png')
        image.save(reference_image_path)
        return image, image_name
   
    return None, None