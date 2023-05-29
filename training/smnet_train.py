from data.data_model import ScanMatchingDataSet
from torchvision.transforms import transforms
from torch.utils.data import random_split, DataLoader
from utils.config import get_train_config
import torch


def train():
    train_config = get_train_config()
    train_size, val_size, test_size = train_config["TRAIN_SIZE"], train_config["VAL_SIZE"], train_config["TEST_SIZE"]
    shuffle = train_config["SHUFFLE_DATASET"]
    batch_size = train_config["BATCH_SIZE"]

    dataset = ScanMatchingDataSet()
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # define the model


if __name__ == "__main__":
    train()
