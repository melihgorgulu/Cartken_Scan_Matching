
from torch import nn
import torch.nn.functional as F

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