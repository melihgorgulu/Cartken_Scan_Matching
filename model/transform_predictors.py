
from torch import nn
import torch.nn.functional as F

# TODO: check how big is your model

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
    
    
class CombinedImageTransformPredictor(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        _, ch, h, w = input_shape
        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=64, kernel_size=3, padding="same", stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same", stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same", stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same", stride=1)
        self.batchnorm2d_1 = nn.BatchNorm2d(num_features=64)
        self.batchnorm2d_2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2d_3 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2d_4 = nn.BatchNorm2d(num_features=128)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.tanh = nn.Tanh()
        self.fcn1 = nn.Linear(128 * (h//(2**4)) * (w//(2**4)), 128)
        self.fcn2 = nn.Linear(128, 4)
        self.flatten = nn.Flatten()
        

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
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2d(x)
        x = self.batchnorm2d_4(x)
        x = self.flatten(x)
        x = self.fcn1(x)
        x = F.relu(x)
        x = self.fcn2(x)
        x = self.tanh(x)
        return x
    
    

    
class CorrelationMapTransformPredictor(nn.Module):
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
        self.fcn2 = nn.Linear(512, 4)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        
        
        self.batchnorm2d_1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2d_2 = nn.BatchNorm2d(num_features=64)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
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
        #x = self.tanh(x)
        return x