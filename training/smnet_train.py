import os

from data.data_model import ScanMatchingDataSet
from torch.utils.data import random_split, DataLoader
from utils.config import get_train_config, get_data_config, get_stats_config
import torch

from model.networks import BasicSMNetwork
from training.losses import CombinedLoss
from training.trainer import SMNetTrainer

from pathlib import Path
from typing import Dict, List
from utils.io_utils import read_json, save_to_json
from scripts.calculate_train_statistics import calculate_all_stats


def train(update_train_stats=False):

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

    # TODO: Right now we are calculating all stats, change it such that we just use train stats
    if update_train_stats:
        data_config: Dict = get_data_config()
        lbl_path = Path(data_config['LABELS_DIR'])
        labels: List = read_json(lbl_path / "lbl.json")['data']
        stats: Dict = calculate_all_stats(labels)
        save_to_json(stats, os.path.join(os.getenv("CONFIG_DIR"), "statsconfig.json"))

    stats_config = get_stats_config()
    # use these train stats in network

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    # define the model
    model = BasicSMNetwork()
    # loss and optimizer
    criterion = CombinedLoss(transform_w=transform_loss_weight, match_w=match_loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # define the trainer
    experiment_name = "test1"
    logger_kwargs = {'update_step': 1, 'show': True}
    trainer = SMNetTrainer(model, criterion, optimizer, logger_kwargs=logger_kwargs,
                           device=device, train_stats_config=stats_config, experiment_name=experiment_name)
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epoch)
    trainer.save_experiment(experiments_dir=Path("experiments"))
    trainer.save_model(Path(f"trained_models/{experiment_name}.pt"))


if __name__ == "__main__":
    train()
