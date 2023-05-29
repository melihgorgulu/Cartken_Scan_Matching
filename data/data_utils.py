from typing import List
import cv2


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
