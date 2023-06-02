
"""
test if the model can over fit with single data.

with few data, test if network can generalize bit.

"""

from data_model import ScanMatchingDataSet
from torch.utils.data import DataLoader
from training.trainer import SMNetTrainer
from utils.config import get_train_config
import torch
from training.losses import CombinedLoss
from model.networks import BasicSMNetwork
from pathlib import Path
import random
from torch import nn


def test_trainer():
    # data params
    train_config = get_train_config()
    train_size, val_size, test_size = train_config["TRAIN_SIZE"], train_config["VAL_SIZE"], train_config["TEST_SIZE"]
    shuffle = train_config["SHUFFLE_DATASET"]
    l2_loss_weight = train_config["TRANSFORM_WEIGHT"]
    bce_loss_weight = train_config["MATCH_WEIGHT"]

    # training params
    # batch_size = train_config["BATCH_SIZE"]
    batch_size = 32
    device = train_config["DEVICE"]
    lr = train_config["LEARNING_RATE"]
    # epoch = train_config["EPOCH"]
    epoch = 100

    dataset = ScanMatchingDataSet()

    # take only one instance to test if the model can over fit
    train_dataset = torch.utils.data.Subset(dataset, [random.randint(0, len(dataset))])
    val_dataset = torch.utils.data.Subset(dataset, [random.randint(0, len(dataset))])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    logger_kwargs = {'update_step': 1, 'show': True}

    model = BasicSMNetwork()
    criterion = CombinedLoss(transform_w=l2_loss_weight, match_w=bce_loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = SMNetTrainer(model, criterion, optimizer, logger_kwargs=logger_kwargs, device=device)
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epoch)
    trainer.save_experiment(experiments_dir=Path("experiments"))


if __name__ == "__main__":
    test_trainer()
