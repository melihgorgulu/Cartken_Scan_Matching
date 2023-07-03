import os

from data.data_model import ScanMatchingDataSet
from data.transforms import Standardize
from data.data_model import DatasetFromSubset
from torch.utils.data import DataLoader
from utils.config import get_train_config, get_data_config, get_stats_config
import torch

from model.networks import BasicSMNetwork
from training.losses import CombinedLoss
from training.trainer import SMNetTrainer

from pathlib import Path
from typing import Dict, List
from utils.io_utils import read_json, save_to_json
from scripts.calculate_train_statistics import calculate_all_stats
from torch.utils.data.dataset import random_split
from torchvision.transforms import Compose


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
    full_dataset = ScanMatchingDataSet()
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
    transform_train = Compose([Standardize(mean=0.1879, std=0.1834)])  # statistics calculated via using training set
    train_dataset = DatasetFromSubset(train_dataset, transform=transform_train)
    # Use train set statistics to prevent information leakage
    val_dataset = DatasetFromSubset(val_dataset, transform=transform_train)

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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # define the model
    model = BasicSMNetwork()
    # loss and optimizer
    criterion = CombinedLoss(transform_w=transform_loss_weight, match_w=match_loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    # define the trainer
    experiment_name = "test_model_droput_small_cnn_and_fcn_0.2_dropout_in_cnn_wd"
    logger_kwargs = {'update_step': 1, 'show': True}
    trainer = SMNetTrainer(model, criterion, optimizer, logger_kwargs=logger_kwargs,
                           device=device, train_stats_config=stats_config, experiment_name=experiment_name,
                           vis_predictions_every_n=1)
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epoch)
    trainer.save_experiment(experiments_dir=Path("experiments"))
    trainer.save_model(Path(f"trained_models"), name=experiment_name)


if __name__ == "__main__":
    train()
