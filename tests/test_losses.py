from training.losses import CombinedLoss
from data_model import ScanMatchingDataSet
from torch.utils.data import DataLoader, random_split
from model.networks import BasicSMNetwork
from utils.config import get_train_config
import torch


train_config = get_train_config()
train_size, val_size, test_size = train_config["TRAIN_SIZE"], train_config["VAL_SIZE"], train_config["TEST_SIZE"]
shuffle = train_config["SHUFFLE_DATASET"]
batch_size = train_config["BATCH_SIZE"]


def test_combined_loss():
    model = BasicSMNetwork()
    # dummy input
    dataset = ScanMatchingDataSet()
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    for cur_data in train_loader:
        cur_img_batch, cur_trans_img_batch, cur_gt_match_batch, cur_gt_trans_batch = cur_data
        prediction = model(cur_img_batch, cur_trans_img_batch)
        gt = (cur_gt_match_batch, cur_gt_trans_batch)
        criterion = CombinedLoss()
        loss, loss_info = criterion(prediction, gt)
        print("Calculated loss: ")
        print(loss.item())
        break


if __name__ == "__main__":
    test_combined_loss()
