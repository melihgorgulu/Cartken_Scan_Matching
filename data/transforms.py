from torch.utils.data.dataset import Dataset
from typing import Any, Optional, List
import torchvision.transforms as T
import torch

class Standardize:
    def __init__(self, dataset: Optional[Dataset] = None, mean=None, std=None):
        # calculate mean and std statistics for standardization
        if dataset:
            self.mean = 0.0
            self.std = 0.0
            for data in dataset:
                img, _, _, _ = data
                cur_mean = img.mean().item()
                cur_std = img.std().item()

                self.mean += cur_mean
                self.std += cur_std

            n = len(dataset)
            self.mean = self.mean / n
            self.std = self.std / n
        elif mean and std:
            self.mean = mean
            self.std = std
        else:
            raise (AssertionError('Please provide either dataset or mean and std'))

    def __call__(self, x):
        return (x - self.mean) / self.std
    

class ResNet50_Transforms:
    # TODO: Add transformation for resnet50
    """_summary_
    Apply required transformations for resnet50
    """
    def __init__(self, h: int ,w: int, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std
        self.h = h
        self.w = w
    
    def __call__(self, x):
        # resize
        transform_resize = T.Resize(size = (self.h, self.w))
        x = transform_resize(x)
        # add rgb channel
        x = torch.cat([x, x, x], dim=0)
        # normalize channelwise
        x[0, ... ] = (x[0,...] - self.mean[0]) / self.std[0]
        x[1, ... ] = (x[1,...] - self.mean[1]) / self.std[1]
        x[2, ... ] = (x[2,...] - self.mean[2]) / self.std[2]
        return x
        
        
        
        
