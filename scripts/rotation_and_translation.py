from pathlib import Path

import torch

from utils.io_utils import convert_pil_to_tensor, show_tensor_image
from PIL import Image
from torch.nn.functional import pad
from torchvision import transforms
from torchvision.transforms.functional import affine
import math


def revert_image_transform(img, transformation):
    cost, sint, tx, ty = transformation

    # calculate radian
    deg_cos = math.degrees(math.acos(cost))
    deg_sin = math.degrees(math.asin(sint))

    deg = None
    if deg_cos > 0 and deg_sin > 0:
        deg = deg_cos
    elif deg_sin < 0:
        deg = -deg_cos

    deg_inv = -deg
    translation = (tx,ty)
    translate_inv = list(map(lambda x: -1*x, list(translations)))
    # first translate back
    reverted_img = affine(img, angle=0, translate=translate_inv, scale=1.0, shear=[0.0, 0.0])
    # then rotate
    reverted_img = affine(reverted_img, angle=deg_inv, translate=[0, 0], scale=1.0, shear=[0.0, 0.0])
    return reverted_img


image_path = Path("/Users/melihgorgulu/Desktop/lenna.png")
img = Image.open(image_path)
img = convert_pil_to_tensor(img, mode="rgb")

# pad image
p1d = (200, 200, 200, 200)
img = pad(img, p1d, "constant", 0)
c, h, w = img.shape

h_t, v_t = 0.2, 0.2
deg_max = 180
deg_min = -180
# select random rotation and translation

degree, translations, scale, shear = transforms.RandomAffine.get_params(degrees=[deg_min, deg_max],
                                                                        translate=[h_t, v_t],
                                                                        scale_ranges=None, shears=None,
                                                                        img_size=[h, w])


trans_img = affine(img, angle=degree, translate=list(translations), scale=scale, shear=list(shear))

# to simulate our case, lets use sin and cos

rad = math.radians(degree)
print("Original degree", degree)

tx, ty = list(translations)
cost, sint = math.cos(rad), math.sin(rad)

gt_transformation = torch.tensor([cost, sint, tx, ty], dtype=torch.float32)

reverted_img = revert_image_transform(trans_img, gt_transformation)

show_tensor_image(img)
show_tensor_image(trans_img)
show_tensor_image(reverted_img)

"""
# get original image
translate_inv = list(map(lambda x: -1*x, list(translations)))
degree_inv = -degree

reverted_img = affine(trans_img, angle=degree_inv, translate=translate_inv, scale=scale, shear=list(shear))


show_tensor_image(img)
show_tensor_image(trans_img)
show_tensor_image(reverted_img)

print(img.shape)
print(trans_img.shape)
print(reverted_img.shape)

"""
