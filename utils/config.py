import os
from utils.io_utils import read_json
from pathlib import Path


def get_data_config() -> dict:
    data_config_path = Path(os.getenv("CONFIG_DIR")) / "dataconfig.json"
    data_config = read_json(data_config_path)
    return data_config


def get_train_config() -> dict:
    train_config_path = Path(os.getenv("CONFIG_DIR")) / "trainconfig.json"
    train_config = read_json(train_config_path)
    return train_config


def get_stats_config() -> dict:
    stats_config_path = Path(os.getenv("CONFIG_DIR")) / "statsconfig.json"
    stats_config = read_json(stats_config_path)
    return stats_config
