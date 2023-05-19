from data.data_model import ScanMatchingDataSet
from utils.io_utils import show_tensor_image
from pathlib import Path
import random
import torch
from torch.utils.data import random_split, DataLoader


def test_dataset():
    sm = ScanMatchingDataSet()
    img, trans_img, lbl = sm[random.randint(0, len(sm))]
    show_tensor_image(img)
    show_tensor_image(trans_img)
    print(lbl)
    print(len(sm))


def test_dataloader():
    sm = ScanMatchingDataSet()
    train_size = 0.7
    test_size = 0.1
    val_size = 0.2

    gen = torch.Generator().manual_seed(42)
    train_data, val_data, test_data = random_split(sm, [train_size, val_size, test_size], generator=gen)
    train_loader = DataLoader(train_data)
    val_loader = DataLoader(val_data)
    test_loader = DataLoader(test_data)
    print(len(train_loader), len(val_loader), len(test_loader))
    for idx, item in enumerate(train_loader):
        img, trans_im, lbl = item
        print(img.shape, trans_im.shape, lbl)
        if idx == 3:
            break


if __name__ == "__main__":
    test_dataset()
    test_dataloader()
