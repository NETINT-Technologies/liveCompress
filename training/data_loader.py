from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from pathlib import Path
from utils import read_file_list, CustomImageDataset
from config import TrainingConfig
import random
import os

class DataModule:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def setup(self):
        # Read train and test folders from file lists
        train_folders = read_file_list(os.path.join(self.config.dataset, 'tri_trainlist.txt'))
        test_folders = read_file_list(os.path.join(self.config.dataset, 'tri_testlist.txt'))
        
        # Limit the number of folders as specified in config
        train_folders = train_folders[:self.config.train_size]
        test_folders = test_folders[:self.config.test_size]

        #perform random shuffling and transforms on train folders
        random.shuffle(train_folders)
        train_folders = train_folders[:self.config.train_size]

        self.transform = transforms.Compose([
            transforms.RandomCrop(self.config.patch_size),
            transforms.ToTensor()
        ])

        self.train_dataset = CustomImageDataset(
            self.config.dataset, 
            train_folders, 
            transform=self.transform
        )
        self.test_dataset = CustomImageDataset(
            self.config.dataset, 
            test_folders, 
            transform=self.transform
        )
    
    def get_train_dataloader(self, rank: int) -> DataLoader:
        sampler = DistributedSampler(self.train_dataset, rank=rank,seed=random.randint(0,1000),shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            sampler=sampler,
            pin_memory=True,
        )

    def get_test_dataloader(self, rank: int) -> DataLoader:
        sampler = DistributedSampler(self.test_dataset, rank=rank,seed=random.randint(0,1000),shuffle=True)
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            num_workers=self.config.num_workers,
            sampler=sampler,
            pin_memory=True,
        )
