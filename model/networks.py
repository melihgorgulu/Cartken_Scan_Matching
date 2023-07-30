
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

def feature_l2_norm(f: torch.Tensor) -> torch.Tensor:
    breakpoint()
    epsilon = 1e-6
    norm = (f.norm(dim=1, keepdim=True) + epsilon)
    
    return f / norm


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
    def __init__(self, backbone_output_shape):
        super().__init__()
        _, _, h , w = backbone_output_shape
        n_in_channel = h*w
        self.conv1 = nn.Conv2d(in_channels=n_in_channel, out_channels=128, kernel_size=3, padding="same", stride=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same", stride=1)
        self.fcn1 = nn.Linear(64*9*9, 512) # TODO: here is hardcoded, change it
        self.fcn2 = nn.Linear(512, 1)
        self.flatten = nn.Flatten()
        
        
        self.batchnorm2d_1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2d_2 = nn.BatchNorm2d(num_features=64)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.fcn2 = nn.Linear(32, 1)

    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool2d(x)
        x = self.batchnorm2d_1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2d(x)
        x = self.batchnorm2d_2(x)
        
        x = self.flatten(x)
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
    def __init__(self, input_shape):
        super().__init__()
        _, ch, h, w = input_shape
        self.fcn1 = nn.Linear(2*ch * h * w, 64)
        self.fcn2 = nn.Linear(64, 32)
        self.fcn3 = nn.Linear(32, 4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fcn1(x)
        x = F.relu(x)
        x = self.fcn2(x)
        x = F.relu(x)
        x = self.fcn3(x)  # Predict 4 values (cosx,sinx,tx,ty)
        x = self.tanh(x)
        return x


class BasicSMNetwork(nn.Module):
    def __init__(self, norm_feature: bool =True):
        super().__init__()
        self.norm_feature = norm_feature
        self.backbone = BasicBackbone()
        backbone_output_shape = self.backbone.get_output_shape()
        self.transform_head = TransformPredictorHead(backbone_output_shape)
        self.feature_matcher_head = FeatureMatcherHead(backbone_output_shape)
        self.flatten = nn.Flatten()


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # extract features
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        # Transform prediction
        f = concat([f1, f2], dim=1)
        f_flattened = self.flatten(f)
        # Transform predictor head
        transform_pred = self.transform_head(f_flattened)
        # Feature matcher head
        # TODO: Another way to calculate corr matrix, also try it out
        # reshape feature maps for matrix multiplication
        b,_,h1,w1 = f1.size()
        b,_,h2,w2 = f2.size()
        if self.norm_feature:
            f1 = feature_l2_norm(f1)
            f2 = feature_l2_norm(f2)
        #f1 = f1.view(b,c,h1*w1).transpose(1,2) # size [b,c,h*w]
        #f2 = f2.view(b,c,h2*w2) # size [b,c,h*w]
        # perform matrix mult.
        #correlation_map = torch.bmm(f1,f2)
        # correlation_map = correlation_map.view(b,h1,w1,h2,w2).unsqueeze(1)
        # With EINSUM
        correlation_map = torch.einsum("bfhw,bfxy->bwhxy", f1, f2)
        correlation_map = correlation_map.view(b, h1*w1, h2, w2)
        if self.norm_feature:
            correlation_map = feature_l2_norm(F.relu(correlation_map))
        match_prob = self.feature_matcher_head(correlation_map)

        return match_prob, transform_pred
