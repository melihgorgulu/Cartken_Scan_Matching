import torch
from torch.utils.data import Dataset
from utils.io_utils import read_json, load_to_tensor
from pathlib import Path
from typing import List, Dict, Tuple
from utils.config import get_data_config
import random
from torchvision.transforms import transforms


class ScanMatchingDataSet(Dataset):

    def __init__(self, return_matched_data_prob: int = 0.6):
        data_config = get_data_config()
        self.data_dir = Path(data_config['DATA_ROOT_DIR'])
        self.lbl_path = Path(data_config['LABELS_DIR'])
        self.labels: List = read_json(self.lbl_path / "lbl.json")['data']
        self.prob = return_matched_data_prob

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param item_idx: Index of item to be fetched
        :return: image tensor, translated image tensor, gt tensor
        """
        choice = random.choices([1, 0], weights=[0.6, 0.4])[0]
        if choice == 1:
            cur_item_lbl: Dict = self.labels[item_idx]
            cur_im_path = self.data_dir / "patches" / cur_item_lbl['org_patch_name']
            cur_trans_img_path = self.data_dir / "transformed_patches" / cur_item_lbl['translated_patch_name']
            cos_gt, sin_gt = cur_item_lbl['gt_rot']
            tx_gt, ty_gt = cur_item_lbl['gt_trans']
            gt_transformation = torch.tensor([cos_gt, sin_gt, tx_gt, ty_gt], dtype=torch.float32)
            gt_is_matched = torch.tensor([1.0], dtype=torch.float32)

            cur_im_tensor = load_to_tensor(cur_im_path)
            cur_trans_tensor = load_to_tensor(cur_trans_img_path)
            # normalize the images
            normalize = transforms.Normalize(mean=torch.mean(cur_im_tensor), std=torch.std(cur_im_tensor))
            cur_im_tensor = normalize(cur_im_tensor)

            normalize = transforms.Normalize(mean=torch.mean(cur_trans_tensor), std=torch.std(cur_trans_tensor))
            cur_trans_tensor = normalize(cur_trans_tensor)

            return cur_im_tensor, cur_trans_tensor, gt_is_matched, gt_transformation
        else:
            cur_item_lbl: Dict = self.labels[item_idx]
            cur_im_path = self.data_dir / "patches" / cur_item_lbl['org_patch_name']
            gt_transformation = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            gt_is_matched = torch.tensor([0.0], dtype=torch.float32)

            # select random transformed image, should be a non-matching scan.
            random_idx = random.choice([i for i in range(len(self.labels)) if i != item_idx])
            nan_match_target_item_lbl: Dict = self.labels[random_idx]
            nan_match_trans_img_path = self.data_dir / "transformed_patches" / nan_match_target_item_lbl[
                'translated_patch_name']

            cur_im_tensor = load_to_tensor(cur_im_path)
            cur_non_match_trans_tensor = load_to_tensor(nan_match_trans_img_path)
            # normalize the images
            normalize = transforms.Normalize(mean=torch.mean(cur_im_tensor), std=torch.std(cur_im_tensor))
            cur_im_tensor = normalize(cur_im_tensor)

            normalize = transforms.Normalize(mean=torch.mean(cur_non_match_trans_tensor),
                                             std=torch.std(cur_non_match_trans_tensor))
            cur_trans_tensor = normalize(cur_non_match_trans_tensor)

            return cur_im_tensor, cur_trans_tensor, gt_is_matched, gt_transformation
