import torch
from torch import nn, concat
import torch.nn.functional as F
from utils.config import get_data_config
from typing import Tuple
import torchvision

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
    
    
class Resnet50(torch.nn.Module):
    """Resnet 50 feature extractor.

    According to [He, Kaiming, et al. Deep residual learning for image recognition].

    .. note::

        Since PyTorch does not provide the feature extractor on its own, we have to create a wrapper
        around the implementation which does not use the final fully connected layer.

    Args:
        pretrained: Whether to download the pretrained weights from PyTorch model store
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self._implementation = torchvision.models.resnet50(pretrained=pretrained, progress=True)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor]:
        """Run the forward pass.

        .. note::

            See https://pytorch.org/vision/stable/models.html on how to normalize the input image:
            ```
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            ```

        Args:
            input: Batch of N3HW normalized input images

        Returns:
            Features from C2 (stride 4), C3 (stride 8), C4 (stride 16), C5 (stride 32) layers
        """
        y = self._implementation.conv1(input)
        y = self._implementation.bn1(y)
        y = self._implementation.relu(y)
        y = self._implementation.maxpool(y)  # 1/2 scale

        c2 = self._implementation.layer1(y)  # 1/4 scale, 256 channels
        c3 = self._implementation.layer2(c2)  # 1/8 scale, 512 channels
        c4 = self._implementation.layer3(c3)  # 1/16 scale, 1024 channels
        c5 = self._implementation.layer4(c4)  # 1/32 scale, 2048 channels

        return c2, c3, c4, c5

    def num_output_channels(self) -> Tuple[int, int, int, int]:
        """Get number of output channels of the output layers.

        Returns:
            Tuple of numbers of output channels in order [C2, C3, C4, C5]
        """
        return (256, 512, 1024, 2048)
    
    def get_output_shape(self) -> Tuple[int, int, int, int]:
        input_shape = get_input_shape()
        x = torch.randn(input_shape)
        x2, x3, x4, x5 = self.forward(x)
        return (x2.shape, x3.shape, x4.shape, x5.shape)
    

