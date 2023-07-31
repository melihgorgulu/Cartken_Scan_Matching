from torch.utils.data.dataset import Dataset
from typing import Optional


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
    pass
