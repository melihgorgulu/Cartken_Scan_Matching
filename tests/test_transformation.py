from pathlib import Path
import json
import random
import torch
from PIL import Image
from utils.io_utils import show_tensor_image, convert_pil_to_tensor
import numpy as np
import cv2


def test():
    labels_path = r"/dataset/labels/lbl.json"
    patch_data_path = Path("/dataset/patches")
    transformed_data_path = Path(
        r"/dataset/transformed_patches")
    with open(labels_path, "r") as f:
        labels = json.load(f)
    n = len(labels['data'])
    index = random.randint(0, n)
    cur_data = labels['data'][index]
    org_patch_name = cur_data['org_patch_name']
    translated_patch_name = cur_data['translated_patch_name']

    org_patch_path = patch_data_path / org_patch_name
    translated_patch_path = transformed_data_path / translated_patch_name

    with Image.open(org_patch_path) as cur_img:
        org_img_tensor = convert_pil_to_tensor(cur_img)

    with Image.open(translated_patch_path) as cur_img:
        translated_img_tensor = convert_pil_to_tensor(cur_img)

    cos_t, sin_t = cur_data['gt_rot']
    rot_matrix = torch.Tensor([[cos_t, -sin_t], [sin_t, cos_t]])
    tx, ty = cur_data['gt_trans']
    inv_rot_matrix = rot_matrix.T
    # apply inv translation
    translation_matrix = np.float32([[1, 0, -tx], [0, 1, -ty]])
    # Apply the translation
    translated_image = cv2.warpAffine(translated_img_tensor[0, :, :].numpy(), translation_matrix, (512, 512))
    inv_rot_matrix = np.float32([[inv_rot_matrix[0][0].item(),
                                  inv_rot_matrix[0][1].item(), 0],
                                 [inv_rot_matrix[1][0].item(),
                                  inv_rot_matrix[1][1].item(), 0]])

    # apply inverse rotation
    output_image = cv2.warpAffine(translated_image, inv_rot_matrix, (512, 512))
    output_image = np.expand_dims(output_image, axis=0)
    output_image = torch.from_numpy(output_image)
    show_tensor_image(org_img_tensor)
    show_tensor_image(translated_img_tensor)
    show_tensor_image(output_image)


if __name__ == "__main__":
    test()
