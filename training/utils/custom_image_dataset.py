from torch.utils.data import Dataset
import os
from PIL import Image

"""
Custom Image Dataset for loading images from a folder structure. Only loads the first image in each folder. 
"""
class CustomImageDataset(Dataset):
    def __init__(self, base_path, folders, transform=None):
        self.base_path = base_path
        self.folders = folders
        self.transform = transform
        self.images = []
        for folder in folders:
            folder_path = os.path.join(base_path, "sequences", folder)
            image_name = os.listdir(folder_path)[0]
            image_path = os.path.join(folder_path, image_name)
            self.images.append(image_path)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image