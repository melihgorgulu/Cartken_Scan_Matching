
import torch
from torch import nn, concat
import torch.nn.functional as F
from utils.config import get_data_config
from typing import Tuple


_data_config = get_data_config()
IMAGE_SIZE = _data_config["IMAGE_HEIGHT"]
BATCH_SIZE = _data_config["BATCH_SIZE"]
N_OF_CH = _data_config["NUMBER_OF_CHANNEL"]

def get_input_shape():
    input_shape = (BATCH_SIZE, N_OF_CH, IMAGE_SIZE, IMAGE_SIZE)
    return input_shape


class BasicBackbone(nn.Module):
    """
    Input: N, C , H , W
    Output: N, 32, H//8, W//8
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=N_OF_CH, out_channels=16, kernel_size=3, padding="same", stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same", stride=1)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same", stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same", stride=1)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same", stride=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same", stride=1)

        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # reduce spatial dimension by half
        self.batchnorm2d_1 = nn.BatchNorm2d(num_features=32)
        self.batchnorm2d_2 = nn.BatchNorm2d(num_features=64)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2d(x)
        x = self.batchnorm2d_1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2d(x)
        x = self.batchnorm2d_2(x)
        
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.maxpool2d(x)
        return x
    
    def get_output_shape(self):
        input_shape = get_input_shape()
        x = torch.randn(input_shape)
        x = self.forward(x)
        return x.shape
        

class FeatureMatcherHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn1 = nn.Linear(256, 32)
        self.fcn2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fcn1(x)
        x = F.relu(x)
        x = self.fcn2(x)
        # We are using BCEWithLogitsLoss, so remove the last sigmoid layer
        # x = F.sigmoid(x)  # probability
        """This loss combines a `Sigmoid` layer and the `BCELoss` in one single class. 
        This version is more numerically stable than using a plain `Sigmoid` followed by a `BCELoss` as, 
        by combining the operations into one layer, we take advantage of the 
        log-sum-exp trick for numerical stability."""

        return x


class TransformPredictorHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = nn.Linear(256, 4)

    def forward(self, x):
        x = self.fcn(x)  # 4 values
        return x


class BasicSMNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform_head = TransformPredictorHead()
        self.feature_matcher_head = FeatureMatcherHead()
        self.backbone = BasicBackbone()
        backbone_output_shape = self.backbone.get_output_shape()
        _, ch, h, w = backbone_output_shape
        self.flatten = nn.Flatten()
        self.fcn1 = nn.Linear(2*ch * h * w, 512)
        self.fcn2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # extract features
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        # Middle
        f = concat([f1, f2], dim=1)
        f_flattened = self.flatten(f)
        x = self.fcn1(f_flattened)
        x = F.relu(x)
        x = self.fcn2(x)
        x = F.relu(x)
        # Feature matcher head
        match_prob = self.feature_matcher_head(x)
        # Transform predictor head
        transform_pred = self.transform_head(x)
        return match_prob, transform_pred
