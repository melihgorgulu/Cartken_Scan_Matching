from data_model import ScanMatchingDataSet
from torch.utils.data import DataLoader, random_split
from training.trainer import SMNetTrainer
from utils.config import get_train_config
import torch
from training.losses import CombinedLoss
from model.networks import BasicSMNetwork


def test_trainer():
    # data params
    train_config = get_train_config()
    train_size, val_size, test_size = train_config["TRAIN_SIZE"], train_config["VAL_SIZE"], train_config["TEST_SIZE"]
    shuffle = train_config["SHUFFLE_DATASET"]
    l2_loss_weight = train_config["TRANSFORM_WEIGHT"]
    bce_loss_weight = train_config["MATCH_WEIGHT"]

    # training params
    # batch_size = train_config["BATCH_SIZE"]
    batch_size = 499
    device = train_config["DEVICE"]
    lr = train_config["LEARNING_RATE"]
    # epoch = train_config["EPOCH"]
    epoch = 3

    dataset = ScanMatchingDataSet()
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
    # For the sake of testing, use test set, because it smaller, one epoch will take less iteration.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    logger_kwargs = {'update_step': 20, 'show': True}

    model = BasicSMNetwork()
    criterion = CombinedLoss(transform_w=l2_loss_weight, match_w=bce_loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = SMNetTrainer(model, criterion, optimizer, logger_kwargs=logger_kwargs, device=device)
    trainer.fit(train_loader=test_loader, val_loader=val_loader, epochs=epoch)


if __name__ == "__main__":
    test_trainer()
