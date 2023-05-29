import os
from utils.io_utils import read_json
from pathlib import Path


def get_data_config() -> dict:
    data_config_dir = Path(os.getenv("DATA_CONFIG_PATH"))
    data_config = read_json(data_config_dir)
    return data_config


def get_train_config() -> dict:
    train_config_dir = Path(os.getenv("TRAIN_CONFIG_PATH"))
    train_config = read_json(train_config_dir)
    return train_config
