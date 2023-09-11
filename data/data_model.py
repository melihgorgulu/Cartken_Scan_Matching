import torch
from torch.utils.data import Dataset
from utils.io_utils import read_json, load_to_tensor
from pathlib import Path
from typing import List, Dict, Tuple
from utils.config import get_data_config
import random
import torchvision.transforms as T

# TODO: Check how much is the dataloading consumes the memory
# TODO: 
class ScanMatchingDataSet(Dataset):

    def __init__(self, transform = None):
        data_config = get_data_config()
        self.data_dir = Path(data_config['DATA_ROOT_DIR'])
        self.lbl_path = Path(data_config['LABELS_DIR'])
        self.labels: List = read_json(self.lbl_path / "lbl.json")['data']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param item_idx: Index of item to be fetched
        :return: image tensor, translated image tensor, gt tensor
        """
        cur_item_lbl: Dict = self.labels[item_idx]
        cur_im_path = self.data_dir / "patches" / cur_item_lbl['org_patch_name']
        cur_trans_img_path = self.data_dir / "transformed_patches" / cur_item_lbl['translated_patch_name']
        cos_gt, sin_gt = cur_item_lbl['gt_rot']
        tx_gt, ty_gt = cur_item_lbl['gt_trans']
        gt_transformation = torch.tensor([cos_gt, sin_gt, tx_gt, ty_gt], dtype=torch.float32)
        gt_is_matched = torch.tensor([cur_item_lbl['gt_match']], dtype=torch.float32)

        cur_im_tensor = load_to_tensor(cur_im_path)
        cur_trans_tensor = load_to_tensor(cur_trans_img_path)
        if self.transform:
            cur_im_tensor = self.transform(cur_im_tensor)
            cur_trans_tensor = self.transform(cur_trans_tensor)
        
        '''
        if self.use_resnet:
            # make it 3 channel and resize
            transform_resize = T.Resize(size = (224,224))
            # resize tensors
            cur_im_tensor = transform_resize(cur_im_tensor)
            cur_trans_tensor = transform_resize(cur_trans_tensor)
            
            # make it 3 channel
            cur_im_tensor = torch.cat([cur_im_tensor, cur_im_tensor, cur_im_tensor], dim=0) 
            cur_trans_tensor = torch.cat([cur_trans_tensor, cur_trans_tensor, cur_trans_tensor], dim=0) 
        '''

        return cur_im_tensor, cur_trans_tensor, gt_is_matched, gt_transformation


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        im, im_trans, gt_is_matched, gt_trans = self.subset[index]
        if self.transform:
            im = self.transform(im)
            im_trans = self.transform(im_trans)
        return im, im_trans, gt_is_matched, gt_trans

    def __len__(self):
        return len(self.subset)
