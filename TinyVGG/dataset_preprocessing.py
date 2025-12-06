from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from pathlib import Path
from utils.custom_dataset import CustomImageDataset

def dataloader(data_path: Path,
               batch_size: int,
               shuffle: bool,
               num_workers: int,
               transform: transforms):
    
    # defining transform tobe applied on data
    transform = transforms.Compose([
    transforms.Resize(size = (128,128)),
    transforms.ToTensor()
    ])

    # custom dataset 
    dataset = CustomImageDataset(data_path,
                                 transform)
    
    # Set manual seed
    torch.manual_seed(42)

    # DataLoader
    dataloader = DataLoader(dataset,
                            batch_size,
                            shuffle,
                            num_workers = num_workers)
    return dataloader
    