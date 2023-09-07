from torch import nn
import torch.nn.functional as F


class FeatureMatcherHead(nn.Module):
    def __init__(self, backbone_output_shape, resnet_backbone = False):
        super().__init__()
        _, _, h , w = backbone_output_shape
        n_in_channel = h*w
        self.conv1 = nn.Conv2d(in_channels=n_in_channel, out_channels=128, kernel_size=3, padding="same", stride=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same", stride=1)
        if resnet_backbone:
            self.fcn1 = nn.Linear(64*18*18, 512) # TODO: here is hardcoded, change it
        else:
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
