from utils.io_utils import revert_image_transform, convert_tensor_to_pil
from data.data_model import ScanMatchingDataSet
from data.transforms import Standardize, ResNet50_Transforms
from data.data_model import DatasetFromSubset
from torch.utils.data import DataLoader
from utils.config import get_train_config
import torch
import matplotlib.pyplot as plt


from typing import List

from torchvision.transforms import Compose

import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
from torchvision.transforms import Compose

def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def get_train_loader():
    train_config = get_train_config()
    # dataset params
    train_size, val_size, test_size = train_config["TRAIN_SIZE"], train_config["VAL_SIZE"], train_config["TEST_SIZE"]
    shuffle = train_config["SHUFFLE_DATASET"]
    # training params
    batch_size = train_config["BATCH_SIZE"]
    batch_size = 1

    # train val and test split
    full_dataset = ScanMatchingDataSet()
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
    print(f"Train test split. Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}, Test Size: {len(test_dataset)}")
    #transform_train = Compose([Standardize(mean=0.1879, std=0.1834)])  # statistics calculated via using training set
    #transform_train = Compose([ResNet50_Transforms(h = 224 ,w = 224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_train = Compose([ResNet50_Transforms()]) # use default std mean
    # transform_train = Compose([ResNet50_Transforms(h = 224 ,w = 224, mean=[0.1879, 0.1879, 0.1879], std=[0.1834, 0.1834, 0.1834])])
    train_dataset = DatasetFromSubset(train_dataset, transform=transform_train)
    # Use train set statistics to prevent information leakage
    val_dataset = DatasetFromSubset(val_dataset, transform=transform_train)

    # use these train stats in network

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader

def test_transforms():
    train_loader = get_train_loader()
    count = 0
    for data in train_loader:
        img_s, img_d, match, transform = data
        if match.item() != 1.0:
            continue
        count += 1
        img_s = img_s[0,...]
        img_d = img_d[0,...]
        match = match[0,...]
        transform = transform[0,...]
        try:
            img_after_back_transform = revert_image_transform(img=img_d, transformation=transform)
        except:
            breakpoint()
            print("hey")
        source_img_pil = convert_tensor_to_pil(img_s)
        targed_img_pil = convert_tensor_to_pil(img_d)
        img_after_back_transform = convert_tensor_to_pil(img_after_back_transform)
        # save
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(source_img_pil)
        axs[0].axis('off')  # Disable axes
        axs[0].set_title("Source Image")
        # Display the second image on the second subplot
        axs[1].imshow(targed_img_pil)
        axs[1].set_title("Trans Image")
        axs[1].axis('off')  # Disable axes

        # Display the third image on the third subplot
        axs[2].imshow(img_after_back_transform)
        axs[2].set_title("Image After backtransform applied")
        axs[2].axis('off')  # Disable axes

        main_title = f" Gt Match: {match.item():.2f}, trans: {transform}"
        fig.suptitle(main_title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        save_path = f"transforms_test/data_{count}.png"
        plt.savefig(str(save_path))
        if count == 10:
            break
    
    
if __name__ == "__main__":
    test_transforms()