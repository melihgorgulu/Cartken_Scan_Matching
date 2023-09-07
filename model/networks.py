import torch
from torch import nn, concat
import torch.nn.functional as F
from utils.config import get_data_config
from typing import Tuple
from .backbones import BasicBackbone, Resnet50
from .matchers import FeatureMatcherHead
from .transform_predictors import TransformPredictorHead


_data_config = get_data_config()
IMAGE_SIZE = _data_config["IMAGE_HEIGHT"]
BATCH_SIZE = _data_config["BATCH_SIZE"]
N_OF_CH = _data_config["NUMBER_OF_CHANNEL"]

def get_input_shape():
    input_shape = (BATCH_SIZE, N_OF_CH, IMAGE_SIZE, IMAGE_SIZE)
    return input_shape

def feature_l2_norm(f: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-6
    norm = (f.norm(dim=1, keepdim=True) + epsilon)
    return f / norm



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
    

class SmNetwithResNetBackBone(nn.Module):
    def __init__(self, norm_feature: bool = True):
        super().__init__()
        self.norm_feature = norm_feature
        self.backbone = Resnet50(pretrained=True)
        backbone_output_shape = self.backbone.get_output_shape()[0] # we are using c2
        self.transform_head = TransformPredictorHead(backbone_output_shape)
        self.feature_matcher_head = FeatureMatcherHead(backbone_output_shape, resnet_backbone=True)
        self.flatten = nn.Flatten()


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # extract features
        s2, _, _, _ = self.backbone(x1) # source features
        t2, _, _, _ = self.backbone(x2) # target features
        b,_,h1,w1 = s2.size()
        b,_,h2,w2 = t2.size()
        if self.norm_feature:
            s2 = feature_l2_norm(s2)
            t2 = feature_l2_norm(t2)
        # Transform prediction
        f = concat([s2, t2], dim=1)
        f_flattened = self.flatten(f)
        # Transform predictor head
        transform_pred = self.transform_head(f_flattened)
        # Feature matcher head
        # TODO: Another way to calculate corr matrix, also try it out
        # reshape feature maps for matrix multiplication

        #f1 = f1.view(b,c,h1*w1).transpose(1,2) # size [b,c,h*w]
        #f2 = f2.view(b,c,h2*w2) # size [b,c,h*w]
        # perform matrix mult.
        #correlation_map = torch.bmm(f1,f2)
        # correlation_map = correlation_map.view(b,h1,w1,h2,w2).unsqueeze(1)
        # With EINSUM
        correlation_map = torch.einsum("bfhw,bfxy->bwhxy", s2, t2)
        correlation_map = correlation_map.view(b, h1*w1, h2, w2)
        if self.norm_feature:
            correlation_map = feature_l2_norm(F.relu(correlation_map))
        match_prob = self.feature_matcher_head(correlation_map)

        return match_prob, transform_pred
