from torch.utils.data.dataset import Dataset
from typing import Tuple, Optional, List
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
    """_summary_
    Apply required transformations for resnet50
    """
    def __init__(self, resize: Optional[Tuple[int]] = None, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        self.mean = mean
        self.std = std
        self.resize = resize
    
    def __call__(self, x):
        # resize
        # TODO: Maybe you dont need resize check it
        if self.resize:
            h, w = self.resize
            transform_resize = T.Resize(size = (h, w))
            x = transform_resize(x)
        if (len(x.shape) != 3 or len(x.shape) != 4):
            # add rgb channel
            x = torch.cat([x, x, x], dim=0)
        # normalize channelwise
        if self.mean and self.std:
            x[0, ... ] = (x[0,...] - self.mean[0]) / self.std[0]
            x[1, ... ] = (x[1,...] - self.mean[1]) / self.std[1]
            x[2, ... ] = (x[2,...] - self.mean[2]) / self.std[2]
            
        else:
            x -= torch.tensor([[[0.485]], [[0.456]], [[0.406]]], device=x.device)
            x /= torch.tensor([[[0.229]], [[0.224]], [[0.225]]], device=x.device)
        return x
        
        
        
        
