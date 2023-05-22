from torch import nn, concat
import torch.nn.functional as F
from utils.config import get_data_config

_data_config = get_data_config()
IMAGE_SIZE = _data_config["IMAGE_HEIGHT"]
BATCH_SIZE = _data_config["BATCH_SIZE"]
N_OF_CH = _data_config["NUMBER_OF_CHANNEL"]


class BasicBackbone(nn.Module):
    """
    Input: N, C , H , W
    Output: N, 32, H//8, W//8
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=N_OF_CH, out_channels=8,
                               kernel_size=3, padding="same", stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding="same", stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same", stride=1)

        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # reduce spatial dimension by half
        self.batchnorm2d_1 = nn.BatchNorm2d(num_features=8)
        self.batchnorm2d_2 = nn.BatchNorm2d(num_features=16)
        self.batchnorm2d_3 = nn.BatchNorm2d(num_features=32)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool2d(x)
        x = self.batchnorm2d_1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2d(x)
        x = self.batchnorm2d_2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool2d(x)
        x = self.batchnorm2d_3(x)

        return x


class FeatureMatcherHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fcn(x)
        x = F.sigmoid(x)  # probability
        return x


class TransformPredictorHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = nn.Linear(512, 4)

    def forward(self, x):
        x = self.fcn(x)
        x = F.relu(x)  # 4 values
        return x


class BasicSMNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform_head = TransformPredictorHead()
        self.feature_matcher_head = FeatureMatcherHead()
        self.backbone = BasicBackbone()
        self.flatten = nn.Flatten()
        self.fcn1 = nn.Linear(64 * 37 * 37, 1024)
        self.fcn2 = nn.Linear(1024, 512)

    def forward(self, x1, x2):
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
