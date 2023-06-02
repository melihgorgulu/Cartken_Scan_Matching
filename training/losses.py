
import torch.nn as nn
from typing import Tuple


class CombinedLoss(nn.Module):
    def __init__(self, criterion_rotation: nn.Module = nn.HuberLoss, criterion_translation: nn.Module = nn.L1Loss,
                 criterion_match: nn.Module = nn.BCEWithLogitsLoss, transform_w: float = 0.6, match_w: float = 0.4):
        super().__init__()
        self.criterion_rotation = criterion_rotation()
        self.criterion_translation = criterion_translation()
        self.criterion_match = criterion_match()
        self.wt = transform_w
        self.wm = match_w

    def forward(self, pred: Tuple, gt: Tuple):
        gt_match, gt_transform = gt
        pred_match, pred_transform = pred
        # TODO: Dont  calculate rotation and translation loss if the pairs are not matched.
        # DONE : Clamp the predicted values for BCE

        '''
        Note: For matching, we are using sigmoid activation function. If we input high negative value to the sigmoid,
        we got 0, and if we give high positive value to the sigmoid, we got 1. This is a problem because since we 
        use logarithm inside the bce, sometimes we have infinitive because of the log(0). To prevent that, we need to 
        clamp the values.
        '''
        # TODO: Also investigate why the input to the sigmoid is too high negative/positive values.
        '''
        One possible reason: since we use combined loss, probably we got big loss because of translation loss, 
        and because of that gradients are also very big, so updating the weight with this exploding gradient gives us
        a bad result. To fix this: We should standardize the target values, such that rotation and translation should be
        in the same scale. Also while combining bce loss and transformation loss, we need to choose weights such that
        at the end our combine loss is meaningful, and not causing exploding gradient/vanishing gradient.
        '''

        # rotation
        pred_rotation, gt_rotation = pred_transform[..., :2], gt_transform[..., :2]

        # translation
        pred_translation, gt_translation = pred_transform[..., 2:], gt_transform[..., 2:]

        binary_match_loss = self.criterion_match(pred_match, gt_match)
        rotation_loss = self.criterion_rotation(pred_rotation, gt_rotation)
        translation_loss = self.criterion_translation(pred_translation, gt_translation)
        transform_loss = rotation_loss + translation_loss
        out = self.wt * transform_loss + self.wm * binary_match_loss

        # return dictionary of losses
        loss_info = {
            "match_loss": binary_match_loss,
            "rotation_loss": rotation_loss,
            "translation_loss": translation_loss,
            "transform_loss": transform_loss,
            "combined_loss": out
        }
        return out, loss_info
