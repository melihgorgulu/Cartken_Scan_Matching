import os

from data.data_model import ScanMatchingDataSet
from data.transforms import Standardize, ResNet50_Transforms
from data.data_model import DatasetFromSubset
from torch.utils.data import DataLoader
from utils.config import get_gridsearch_config, get_data_config, get_stats_config
import torch
import torch.optim as optim


from model.networks import BasicSMNetwork, SmNetwithResNetBackBone, SmNetwithResNetBackBone_Small
from training.losses import CombinedLoss
from training.trainer import SMNetTrainer

from pathlib import Path
from typing import Dict, List
from utils.io_utils import read_json, save_to_json
from scripts.calculate_train_statistics import calculate_all_stats

from torchvision.transforms import Compose
from data.data_utils import random_split

def grid_search():
    gridsearch_config = get_gridsearch_config()
    # dataset params
    train_size, val_size, test_size = gridsearch_config["TRAIN_SIZE"], gridsearch_config["VAL_SIZE"], gridsearch_config["TEST_SIZE"]
    shuffle = gridsearch_config["SHUFFLE_DATASET"]
    # training params
    batch_size = gridsearch_config["BATCH_SIZE"]
    transform_loss_weight: List = gridsearch_config["TRANSFORM_WEIGHT"]
    translation_loss_weight: List = gridsearch_config['TRANSLATION_WEIGHT']
    rotation_loss_weight: List = gridsearch_config['ROTATION_WEIGHT']
    match_loss_weight: List = gridsearch_config["MATCH_WEIGHT"]
    device = gridsearch_config["DEVICE"]
    lr: List = gridsearch_config["LEARNING_RATE"]
    wd: List = gridsearch_config["WEIGHT_DECAY"]
    epoch = gridsearch_config["EPOCH"]
    # train val and test split
    print("______STARTING TO GRID SEARCH______")
    index = 0
    for lr_ in lr:
        for wd_ in wd:
            for transform_w_ in transform_loss_weight:
                for translate_w_ in translation_loss_weight:
                    for rotation_w_ in rotation_loss_weight:
                        for match_w_ in match_loss_weight:
                            print("*"*100)
                            print(f"SEARCH NO: {index+1}")
                            print(f"&CURRENT PARAMS&: lr: {lr_} , wd: {wd_}, transform_w: {transform_w_} , match_w: {match_w_}, rot_w: {rotation_w_} , tl_w: {translate_w_} ")
                            full_dataset = ScanMatchingDataSet()
                            train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                                                    generator=torch.Generator().manual_seed(42))
                            print(f"Train test split. Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}, Test Size: {len(test_dataset)}")
                            #transform_train = Coqmpose([Standardize(mean=0.1879, std=0.1834)])  # statistics calculated via using training set
                            #transform_train = Compose([ResNet50_Transforms(h = 224 ,w = 224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                            transform_train = Compose([ResNet50_Transforms()]) # use default std mean
                            # transform_train = Compose([ResNet50_Transforms(h = 224 ,w = 224, mean=[0.1879, 0.1879, 0.1879], std=[0.1834, 0.1834, 0.1834])])
                            train_dataset = DatasetFromSubset(train_dataset, transform=transform_train)
                            # Use train set statistics to prevent information leakage
                            val_dataset = DatasetFromSubset(val_dataset, transform=transform_train)

                            if gridsearch_config["UPDATE_TRAIN_STATS"]:
                                data_config: Dict = get_data_config()
                                lbl_path = Path(data_config['LABELS_DIR'])
                                labels: List = read_json(lbl_path / "lbl.json")['data']
                                stats: Dict = calculate_all_stats(labels)
                                save_to_json(stats, os.path.join(os.getenv("CONFIG_DIR"), "statsconfig.json"))
                                print("Train stats are updated.")

                            stats_config = get_stats_config()
                            # use these train stats in network

                            # dataloaders

                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                            # define the model
                            # model = BasicSMNetwork()
                            model = SmNetwithResNetBackBone_Small()
                            # loss and optimizer
                            criterion = CombinedLoss(transform_w=transform_w_, match_w=match_w_, 
                                                    translation_w=translate_w_, rotation_w=rotation_w_)
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=wd_)
                            if gridsearch_config["USE_SCHEDULER"]:
                                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=gridsearch_config["SCHEDULER_STEP_SIZE"], 
                                                                    gamma=gridsearch_config["SCHEDULER_GAMMA"])
                            else:
                                scheduler = None
                            # define the trainer
                            experiment_name = f"22_09_2023_gridsearch_{index}_lr{lr_:.2f}_wd_{wd_:.2f}_small"
                            index += 1 
                            logger_kwargs = {'update_step': 1, 'show': True}
                            trainer = SMNetTrainer(model, criterion, optimizer, logger_kwargs=logger_kwargs,
                                                device=device, train_stats_config=stats_config, experiment_name=experiment_name,
                                                vis_predictions_every_n=None, show_all_losses=True, use_early_stop=gridsearch_config["USE_EARLYSTOP"], scheduler=scheduler, is_grid_search=True)
                            trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=epoch)
                            trainer.save_experiment(experiments_dir=Path("experiments"))
                            model_save_path = Path(f"trained_models")
                            if not model_save_path.exists():
                                model_save_path.mkdir()
                            trainer.save_model(model_save_path, name=experiment_name)
                            print("*"*100)

    print("______END OF GRID SEARCH______")

if __name__ == "__main__":
    grid_search()
