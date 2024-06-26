
import random
from pathlib import Path
from typing import Dict
from data_preprocessing import *
import torch
from data_utils import *
from utils.io_utils import *

import math
import logging

logger = logging.getLogger()


def create_dataset(meta_data_path: Path):
    dataset_path = Path("dataset")
    patches_dir = dataset_path / "patches"
    transformed_patches_dir = dataset_path / "transformed_patches"
    labels_dir = dataset_path / "labels"
    if not dataset_path.exists():
        dataset_path.mkdir()
        patches_dir.mkdir()
        transformed_patches_dir.mkdir()
        labels_dir.mkdir()

    root_data_path = Path("/home/melihgorgulu/cartken/Cartken_Scan_Matching/data/dataset/maps")
    gt_dict = {'data': []}
    for meta_data_path in meta_data_path.iterdir():
        if ".DS_Store" in str(meta_data_path):
            continue
        rectangle_data_dict = read_json(meta_data_path)
        map_name = str(meta_data_path.name).replace(".json", "")
        map_image_path = root_data_path / map_name / "map.png"
        rotated_cropper = RandomRotatedCrop(show_generated_rectangle=False)
        map_image = load_to_tensor(Path(map_image_path))
        map_image = torch.squeeze(map_image, dim=0)

        index = 0
        print(f"________ Processing map -> {map_name}________")
        for cur_key_rect in rectangle_data_dict['data']:
            cur_key_rect_id = cur_key_rect['id']
            cur_key_rect_obj = RectangleCrop(center=tuple(cur_key_rect['center']), angle=cur_key_rect['angle'],
                                             crop_size=(cur_key_rect['H'], cur_key_rect['W']))

            cur_key_img = rotated_cropper(map_image, cur_key_rect_obj)
            cur_key_img_name = f"{map_name}_{cur_key_rect_id}.png"
            save_tensor_as_image(cur_key_img, save_path=patches_dir / cur_key_img_name)
            cur_samp_rect_index = 0
            # create data for matched samples
            for cur_samp_rect in cur_key_rect['matched_samples']:
                cur_samp_rect_obj = RectangleCrop(center=tuple(cur_samp_rect['center']), angle=cur_samp_rect['angle'],
                                                  crop_size=(cur_samp_rect['H'], cur_samp_rect['W']))
                cur_samp_img = rotated_cropper(map_image, cur_samp_rect_obj)
                cur_samp_img_name = f"{map_name}_{cur_key_rect_id}_trans_{cur_samp_rect_index}.png"
                cur_samp_rect_index += 1
                save_tensor_as_image(cur_samp_img, save_path=transformed_patches_dir / cur_samp_img_name)
                # create GT Labels
                # create GT cos and sin
                angle_diff = cur_samp_rect_obj.angle - cur_key_rect_obj.angle
                if angle_diff < 0:
                    angle_diff = 360 + angle_diff
                radian = math.radians(angle_diff)
                gt_rot = [math.cos(radian), math.sin(radian)]
                # create GT tx and ty
                gt_tx = cur_samp_rect_obj.center_x - cur_key_rect_obj.center_x
                gt_ty = cur_samp_rect_obj.center_y - cur_key_rect_obj.center_y
                gt_trans = [gt_tx, gt_ty]
                cur_lbl_dict = {"map_name": map_name,
                                "org_patch_name": Path(cur_key_img_name).name,
                                "translated_patch_name": Path(cur_samp_img_name).name,
                                "data_idx": index,
                                "gt_rot": gt_rot,
                                "gt_trans": gt_trans,
                                "gt_match": 1.0}
                gt_dict['data'].append(cur_lbl_dict)
                print(index)
                index += 1
            # create data for un-matched samples
            cur_un_samp_rect_index = 0
            for cur_un_samp_rect in cur_key_rect['unmatched_samples']:
                cur_un_samp_rect_obj = RectangleCrop(center=tuple(cur_un_samp_rect['center']),
                                                     angle=cur_un_samp_rect['angle'],
                                                     crop_size=(cur_un_samp_rect['H'], cur_un_samp_rect['W']))
                cur_un_samp_img = rotated_cropper(map_image, cur_un_samp_rect_obj)
                cur_un_samp_img_name = f"{map_name}_{cur_key_rect_id}_unmatched_{cur_un_samp_rect_index}.png"
                cur_un_samp_rect_index += 1
                save_tensor_as_image(cur_un_samp_img, save_path=transformed_patches_dir / cur_un_samp_img_name)
                # create GT Labels for unmatched pairs
                gt_rot = [0, 0]
                # create GT tx and ty
                gt_trans = [0, 0]
                cur_lbl_dict = {"org_patch_name": Path(cur_key_img_name).name,
                                "translated_patch_name": Path(cur_un_samp_img_name).name,
                                "data_idx": index,
                                "gt_rot": gt_rot,
                                "gt_trans": gt_trans,
                                "gt_match": 0.0}
                gt_dict['data'].append(cur_lbl_dict)
                print(index)
                index += 1

    # shuffle gt_dict
    random.shuffle(gt_dict['data'])
    save_to_json(dictionary=gt_dict, json_path=str(labels_dir / "lbl.json"))


if __name__ == "__main__":
    meta_data_path = Path("/home/melihgorgulu/cartken/Cartken_Scan_Matching/data/dataset/data_meta")
    create_dataset(meta_data_path)

    print("DONE!!")
