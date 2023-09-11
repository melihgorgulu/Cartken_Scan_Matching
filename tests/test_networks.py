from model.networks import BasicSMNetwork, BasicBackbone, SmNetwithResNetBackBone
import torch
from utils.config import get_data_config

_data_config = get_data_config()
IMAGE_SIZE = _data_config["IMAGE_HEIGHT"]
BATCH_SIZE = _data_config["BATCH_SIZE"]
N_OF_CH = _data_config["NUMBER_OF_CHANNEL"]


def get_input_shape():
    input_shape = (BATCH_SIZE, N_OF_CH, IMAGE_SIZE, IMAGE_SIZE)
    return input_shape


def test_backbone():
    model = BasicBackbone()
    input_shape = get_input_shape()
    x = torch.randn(input_shape)
    print(f'Input shape {x.shape}')
    x = model(x)
    print(f'Output shape {x.shape}')


def test_model():
    # model = BasicSMNetwork()
    model = SmNetwithResNetBackBone()
    input_shape = get_input_shape()
    print(f'Input: Two image with shape {input_shape}')
    x1 = torch.randn(input_shape)
    x2 = torch.randn(input_shape)
    match_prob, transform_pred = model(x1, x2)
    print(match_prob.shape, transform_pred.shape)


if __name__ == "__main__":
    test_model()
