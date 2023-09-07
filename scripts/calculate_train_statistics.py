
from utils.config import get_data_config
from pathlib import Path
from typing import List, Dict
from utils.io_utils import read_json
import numpy as np


def get_stats(data: np.array) -> Dict:
    mean = float(np.mean(data))
    std = float(np.std(data))
    max_ = float(np.max(data))
    min_ = float(np.min(data))
    out = {
        "mean": mean,
        "std": std,
        "max": max_,
        "min": min_
    }
    return out


def calculate_all_stats(labels_list: List) -> Dict:
    # rotation statistics
    rot_cos_all = np.array([i["gt_rot"][0] for i in labels_list])
    rot_sin_all = np.array([i["gt_rot"][1] for i in labels_list])
    rot_cos_stats = get_stats(rot_cos_all)
    rot_sin_stats = get_stats(rot_sin_all)
    # -----O------
    # translation statistics
    translation_x_all = np.array([i["gt_trans"][0] for i in labels_list])
    translation_y_all = np.array([i["gt_trans"][1] for i in labels_list])

    tx_stats = get_stats(translation_x_all)
    ty_stats = get_stats(translation_y_all)
    # -----O------
    # match statistics
    number_of_match = 0
    for i in labels_list:
        number_of_match += int(i["gt_match"])
        
    number_of_unmatch = len(labels_list) - number_of_match
        

    out = {
        "dataset_size": len(labels_list),
        "cos_stats": rot_cos_stats,
        "sin_stats": rot_sin_stats,
        "tx_stats": tx_stats,
        "ty_stats": ty_stats,
        "number_of_match":number_of_match,
        "number_of_unmatch":number_of_unmatch,
    }

    return out


def main():
    data_config: Dict = get_data_config()
    lbl_path = Path(data_config['LABELS_DIR'])
    labels: List = read_json(lbl_path / "lbl.json")['data']
    # for now calculate statistics of all data, but while training make sure to use just training set
    # statistics to prevent information leakage
    print(calculate_all_stats(labels))


if __name__ == "__main__":
    main()

