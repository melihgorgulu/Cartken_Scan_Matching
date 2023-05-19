from model.networks import BasicSMNetwork
import torch

INPUT_SIZE = 300
BATCH_SIZE = 32


def test_model():
    model = BasicSMNetwork()
    input_shape = (BATCH_SIZE, 1, INPUT_SIZE, INPUT_SIZE)  # same size as our input image (batch_size, h,w,d)
    print(f'Input shape {input_shape}')
    x1 = torch.randn(input_shape)
    x2 = torch.randn(input_shape)
    match_logit, pred = model(x1, x2)
    print(match_logit.shape, pred.shape)


test_model()
