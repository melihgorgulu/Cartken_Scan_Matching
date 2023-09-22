from typing import List
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
import warnings
import math

def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

# Ref: https://github.com/obss/sahi/blob/e798c80d6e09079ae07a672c89732dd602fe9001/sahi/slicing.py#L30, MIT License

def calculate_patches(
        image_height: int,
        image_width: int,
        patch_height: int = 512,
        patch_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
) -> List[List[int]]:
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping slices according to the height and width provided. These slices are returned
    as bounding boxes in xyxy format.

    :param image_height: Height of the original image.
    :param image_width: Width of the original image.
    :param patch_height: Height of each slice
    :param patch_width: Width of each slice
    :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :return: a list of bounding boxes in xyxy format
    """

    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * patch_height)
    x_overlap = int(overlap_width_ratio * patch_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + patch_height
        while x_max < image_width:
            x_max = x_min + patch_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - patch_width)
                ymin = max(0, ymax - patch_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def visualise_patches_on_img(img_path, patches):
    image = cv2.imread(img_path)
    # Create a window to display the image
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    current_index = 0

    # Loop until all bounding boxes have been drawn
    while current_index < len(patches):
        # Draw the current bounding box on the image
        box = patches[current_index]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Display the image and wait for a keypress
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)

        # If the key is 1, move to the next bounding box
        if key == ord("1"):
            current_index += 1

    # Cleanup
    cv2.destroyAllWindows()


def is_inside_rectangle(rect, num_cols: int, num_rows: int) -> bool:
    """ Determine if the four corners of the rectangle are inside the rectangle with width and height
    :param rect: RectangleCrop
    :param num_cols: Other rectangle's # of cols
    :param num_rows: Other rectangle's # of rows
    :return:
        True: if the rotated sub rectangle is side the up-right rectangle
        False: else
    """
    rect_center_x = rect.center_x
    rect_center_y = rect.center_y

    if (rect_center_x < 0) or (rect_center_x > num_cols):
        return False
    if (rect_center_y < 0) or (rect_center_y > num_rows):
        return False

    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    rect_center = (rect_center_x, rect_center_y)
    rect_wh = (rect.crop_w, rect.crop_h)
    angle = rect.angle
    rect_in = (rect_center, rect_wh, angle)
    box = cv2.boxPoints(rect_in)

    x_max = int(np.max(box[:, 0]))
    x_min = int(np.min(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    y_min = int(np.min(box[:, 1]))

    if (x_max <= num_cols) and (x_min >= 0) and (y_max <= num_rows) and (y_min >= 0):
        return True
    else:
        return False


def rect_bbx(rect):
    # Rectangle bounding box for rotated rectangle
    # Example:
    # rotated rectangle: height 4, width 4, center (10, 10), angle 45 degree
    # bounding box for this rotated rectangle, height 4*sqrt(2), width 4*sqrt(2), center (10, 10), angle 0 degree

    rect_center = (rect.center_x, rect.center_y)
    rect_wh = (rect.crop_w, rect.crop_h)
    angle = rect.angle
    rect_in = (rect_center, rect_wh, angle)

    box = cv2.boxPoints(rect_in)
    x_max = int(np.max(box[:, 0]))
    x_min = int(np.min(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    y_min = int(np.min(box[:, 1]))

    # Top-left
    # (x_min, y_min)
    # Top-right
    # (x_min, y_max)
    # Bottom-left
    #  (x_max, y_min)
    # Bottom-right
    # (x_max, y_max)
    # Width
    # y_max - y_min
    # Height
    # x_max - x_min
    # Center
    # (x_min + x_max) // 2, (y_min + y_max) // 2

    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    angle = 0

    return center, (width, height), angle


def image_rotate_without_crop(mat, angle):
    # https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    if type(mat) == torch.Tensor:
        mat = mat.detach().cpu().numpy()
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    return rotated_mat


def crop_rectangle(image, rect):
    # rect has to be upright

    num_rows = image.shape[0]
    num_cols = image.shape[1]

    if not is_inside_rectangle(rect=rect, num_cols=num_cols, num_rows=num_rows):
        print("Proposed rectangle is not fully in the image.")
        return None

    rect_center_x = rect.center_x
    rect_center_y = rect.center_y
    rect_width = rect.crop_w
    rect_height = rect.crop_h

    return image[rect_center_y - rect_height // 2:rect_center_y + rect_height - rect_height // 2,
           rect_center_x - rect_width // 2:rect_center_x + rect_width - rect_width // 2]



def visualize_crop_on_map(img: torch.Tensor, rectangle, f_name: str):
    map_image = img.detach().cpu().numpy()  # torch tensor to numpy array
    rect_center = (rectangle.center_x, rectangle.center_y)
    rect_wh = (rectangle.crop_w, rectangle.crop_h)
    angle = rectangle.angle
    rect_in = (rect_center, rect_wh, angle)
    box = cv2.boxPoints(rect_in).astype(np.int64)

    # to make sure imshow work
    map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(map_image, [box], 0, (255, 0, 0), 10)
    cv2.arrowedLine(map_image, rect_center, ((box[1][0] + box[2][0]) // 2, (box[1][1] + box[2][1]) // 2),
                    (255, 0, 0),
                    3, tipLength=0.2)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(map_image)
    plt.tight_layout()
    plt.savefig(f'{f_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualise_patches_on_img(img_path, patches):
    image = cv2.imread(img_path)
    # Create a window to display the image
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    current_index = 0

    # Loop until all bounding boxes have been drawn
    while current_index < len(patches):
        # Draw the current bounding box on the image
        box = patches[current_index]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Display the image and wait for a keypress
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)

        # If the key is 1, move to the next bounding box
        if key == ord("1"):
            current_index += 1

    # Cleanup
    cv2.destroyAllWindows()


def visualize_crops_on_map(image: torch.Tensor, rectangles):
    image = image.permute(1, 2, 0) # take channel dim to the last
    image = image.detach().cpu().numpy()  # torch tensor to numpy array
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Create a window to display the image
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    current_index = 0

    # Loop until all bounding boxes have been drawn
    while current_index < len(rectangles):
        # Draw the current bounding box on the image
        rectangle = rectangles[current_index]
        rect_center = (rectangle.center_x, rectangle.center_y)
        rect_wh = (rectangle.crop_w, rectangle.crop_h)
        angle = rectangle.angle
        rect_in = (rect_center, rect_wh, angle)
        box = cv2.boxPoints(rect_in).astype(np.int64)
        cv2.drawContours(image, [box], 0, (255, 0, 0), 10)
        cv2.arrowedLine(image, rect_center, ((box[1][0] + box[2][0]) // 2, (box[1][1] + box[2][1]) // 2),
                        (255, 0, 0),
                        3, tipLength=0.2)


        # Display the image and wait for a keypress
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)

        # If the key is 1, move to the next bounding box
        if key == ord("1"):
            current_index += 1

    # Cleanup
    cv2.destroyAllWindows()


def calculate_patches(
        image_height: int,
        image_width: int,
        patch_height: int = 512,
        patch_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
) -> List[List[int]]:
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping slices according to the height and width provided. These slices are returned
    as bounding boxes in xyxy format.

    :param image_height: Height of the original image.
    :param image_width: Width of the original image.
    :param patch_height: Height of each slice
    :param patch_width: Width of each slice
    :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :return: a list of bounding boxes in xyxy format
    """

    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * patch_height)
    x_overlap = int(overlap_width_ratio * patch_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + patch_height
        while x_max < image_width:
            x_max = x_min + patch_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - patch_width)
                ymin = max(0, ymax - patch_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def filter_patches(image: torch.Tensor, patches, p_h, p_w, threshold: float = 0.5):
    index_to_exclude = []
    for cur_idx, cur_patch_coord in enumerate(patches):
        x1, y1, x2, y2 = cur_patch_coord[0], cur_patch_coord[1], cur_patch_coord[2], cur_patch_coord[3]
        cur_patch = image[:, y1:y2, x1:x2]
        if cur_patch.shape[1] != p_h or cur_patch.shape[2] != p_w:
            index_to_exclude.append(cur_idx)
            continue
        _c, _h, _w = cur_patch.size()
        n_of_pixel = _c * _h * _w
        n_of_white_pixel = torch.count_nonzero(cur_patch)
        white_pixel_rate = torch.div(n_of_white_pixel, n_of_pixel)
        if white_pixel_rate < threshold:
            index_to_exclude.append(cur_idx)
    filtered_patches = [_p for _i, _p in enumerate(patches) if _i not in index_to_exclude]
    return filtered_patches