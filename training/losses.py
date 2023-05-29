import torch
import torch.nn as nn
from typing import Tuple


class CombinedLoss(nn.Module):
    def __init__(self, criterion_transform: nn.Module = nn.MSELoss, criterion_match: nn.Module = nn.BCELoss,
                 transform_w: float = 0.6, match_w: float = 0.4):
        super().__init__()
        self.criterion_transform = criterion_transform()
        self.criterion_match = criterion_match()
        self.wt = transform_w
        self.wm = match_w

    def forward(self, pred: Tuple, gt: Tuple):
        gt_match, gt_trans = gt
        pred_match, pred_trans = pred
        match_loss = self.criterion_match(pred_match, gt_match)
        transform_loss = self.criterion_transform(pred_trans, gt_trans)
        out = self.wt * transform_loss + self.wm * match_loss
        return out
