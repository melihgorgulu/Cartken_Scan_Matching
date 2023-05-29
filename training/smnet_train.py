from data.data_model import ScanMatchingDataSet
from torchvision.transforms import transforms
from torch.utils.data import random_split, DataLoader
from utils.config import get_train_config
import torch
from model.networks import BasicSMNetwork
from training.losses import CombinedLoss
from training.trainer import SMNetTrainer
from pathlib import Path


def train():
    train_config = get_train_config()
    # dataset params
    train_size, val_size, test_size = train_config["TRAIN_SIZE"], train_config["VAL_SIZE"], train_config["TEST_SIZE"]
    shuffle = train_config["SHUFFLE_DATASET"]
    # training params
    batch_size = train_config["BATCH_SIZE"]
    transform_loss_weight = train_config["TRANSFORM_WEIGHT"]
    match_loss_weight = train_config["MATCH_WEIGHT"]
    device = train_config["DEVICE"]
    lr = train_config["LEARNING_RATE"]
    epoch = train_config["EPOCH"]

    # train val and test split
    dataset = ScanMatchingDataSet()
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # define the model
    model = BasicSMNetwork()
    # loss and optimizer
    criterion = CombinedLoss(transform_loss_weight, match_loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # define the trainer
    logger_kwargs = {'update_step': 20, 'show': True}
    trainer = SMNetTrainer(model, criterion, optimizer, logger_kwargs=logger_kwargs, device=device)
    trainer.fit(train_loader=test_loader, val_loader=val_loader, epochs=epoch)
    trainer.save_experiment(experiments_dir=Path("experiments"))


if __name__ == "__main__":
    train()
