
"""
test if the model can over fit with single data.

with few data, test if network can generalize bit.

"""

import os

from data.data_model import ScanMatchingDataSet
from data.transforms import Standardize, ResNet50_Transforms
from data.data_model import DatasetFromSubset
from torch.utils.data import DataLoader
from utils.config import get_train_config, get_data_config, get_stats_config
import torch
import torch.optim as optim

from model.networks import BasicSMNetwork, SmNetwithResNetBackBone, SmNetCorrResNetBackBone, SmNetwithResNetBackBone_Small
from training.losses import CombinedLoss
from training.trainer import SMNetTrainer

from pathlib import Path
from typing import Dict, List
from utils.io_utils import read_json, save_to_json
from scripts.calculate_train_statistics import calculate_all_stats

from torchvision.transforms import Compose

from data.data_utils import random_split

def test_model_training(update_train_stats=False):
    
    train_config = get_train_config()
    # dataset params
    train_size, val_size, test_size = train_config["TRAIN_SIZE"], train_config["VAL_SIZE"], train_config["TEST_SIZE"]
    shuffle = train_config["SHUFFLE_DATASET"]
    # training params
    batch_size = train_config["BATCH_SIZE"]
    transform_loss_weight = train_config["TRANSFORM_WEIGHT"]
    translation_loss_weight = train_config['TRANSLATION_WEIGHT']
    rotation_loss_weight = train_config['ROTATION_WEIGHT']
    match_loss_weight = train_config["MATCH_WEIGHT"]
    device = train_config["DEVICE"]
    lr = train_config["LEARNING_RATE"]
    wd = train_config["WEIGHT_DECAY"]
    epoch = train_config["EPOCH"]

    # train val and test split
    
    full_dataset = ScanMatchingDataSet()
    train_size = 0.001
    val_size = 0.001
    test_size = 0.998
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

    if train_config["UPDATE_TRAIN_STATS"]:
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
    # model = BasicSMNetwork()
    model = SmNetwithResNetBackBone_Small()
    # loss and optimizer
    criterion = CombinedLoss(transform_w=transform_loss_weight, match_w=match_loss_weight, 
                             translation_w=translation_loss_weight, rotation_w=rotation_loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if train_config["USE_SCHEDULER"]:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_config["SCHEDULER_STEP_SIZE"], 
                                              gamma=train_config["SCHEDULER_GAMMA"])
    else:
        scheduler = None
    # define the trainer
    experiment_name = "test_model_general_small_higher_lr_1e-3"
    logger_kwargs = {'update_step': 1, 'show': True}
    trainer = SMNetTrainer(model, criterion, optimizer, logger_kwargs=logger_kwargs,
                           device=device, train_stats_config=stats_config, experiment_name=experiment_name,
                           vis_predictions_every_n=None, show_all_losses=True, use_early_stop=train_config["USE_EARLYSTOP"], scheduler=scheduler)
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epoch)
    trainer.save_experiment(experiments_dir=Path("experiments"))
    model_save_path = Path(f"trained_models")
    if not model_save_path.exists():
        model_save_path.mkdir()
    trainer.save_model(model_save_path, name=experiment_name)


if __name__ == "__main__":
    test_model_training(update_train_stats=False)
