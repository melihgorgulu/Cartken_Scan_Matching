import os

from numpy import ndarray

from utils.config import get_data_config
from pathlib import Path
from typing import List, Dict, Tuple
from utils.io_utils import read_json
import numpy as np


def calculate_stats(data: np.array):
    mean = np.mean(data, axis=0)
    std = np.std(data)
    max_ = np.max(data)
    min_ = np.min(data)
    return mean, std, max_, min_


def main():
    data_config: Dict = get_data_config()
    lbl_path = Path(data_config['LABELS_DIR'])
    labels: List = read_json(lbl_path / "lbl.json")['data']
    # for now calculate statistics of all data, but while training make sure to use just training set
    # statistics to prevent information leakage
    translation_x_all = np.array([i["gt_trans"][0] for i in labels])
    translation_y_all = np.array([i["gt_trans"][1] for i in labels])
    print("tx stats (mean,std, max, min): ", calculate_stats(translation_x_all))
    print("ty stats (mean,std, max, min): ", calculate_stats(translation_y_all))


if __name__ == "__main__":
    main()
