from data_preprocessing import *
import matplotlib.pyplot as plt
import random
import math
from typing import Dict, List
from data_utils import crop_rectangle
from utils.io_utils import *

"""
THIS SCRIPT CREATES AUGMENTED DATA DICTIONARY WHICH CAN BE PROCESSED FOR CREATING DATASET.
"""


def visualize_rect_dict_on_map(image: torch.Tensor, rectangles_dict: Dict):
    image = image.permute(1, 2, 0)  # take channel dim to the last
    image = image.detach().cpu().numpy()  # torch tensor to numpy array
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Create a window to display the image
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    current_key_index = 0
    # Loop until all bounding boxes have been drawn
    while current_key_index < len(rectangles_dict['data']):
        # Draw the current bounding box on the image
        cur_key_rect = rectangles_dict['data'][current_key_index]
        rectangle = RectangleCrop(center=tuple(cur_key_rect['center']), angle=cur_key_rect['angle'],
                                  crop_size=(cur_key_rect['H'], cur_key_rect['W']))
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
            current_key_index += 1
        elif key == ord("2"):
            current_sample_index = 0
            while current_sample_index < len(cur_key_rect['matched_samples']):
                # Draw the current bounding box on the image
                cur_sample_rect = cur_key_rect['matched_samples'][current_sample_index]
                rectangle = RectangleCrop(center=tuple(cur_sample_rect['center']), angle=cur_sample_rect['angle'],
                                          crop_size=(cur_sample_rect['H'], cur_sample_rect['W']))
                rect_center = (rectangle.center_x, rectangle.center_y)
                rect_wh = (rectangle.crop_w, rectangle.crop_h)
                angle = rectangle.angle
                rect_in = (rect_center, rect_wh, angle)
                box = cv2.boxPoints(rect_in).astype(np.int64)
                cv2.drawContours(image, [box], 0, (0, 0, 255), 10)
                cv2.arrowedLine(image, rect_center, ((box[1][0] + box[2][0]) // 2, (box[1][1] + box[2][1]) // 2),
                                (0, 0, 255),
                                3, tipLength=0.2)

                # Display the image and wait for a keypress
                cv2.imshow("Image", image)
                key = cv2.waitKey(0)
                if key == ord("2"):
                    current_sample_index += 1
                else:
                    break

    # Cleanup
    cv2.destroyAllWindows()


def sample_n_rectangles(map_image: torch.Tensor, source_rect: RectangleCrop, rect_h, rect_w, n, th=0.4):
    if len(map_image.shape) == 3:
        map_image = torch.squeeze(map_image, dim=0)
        map_h, map_w = map_image.size()
    else:
        map_h, map_w = map_image.size()

    source_center_x = source_rect.center_x
    source_center_y = source_rect.center_y

    sampled_points = sample_points_in_rectangle(rect_w, rect_w, 5 * n)
    sampled_centers = extract_global_centers_coord_from_points((source_center_x, source_center_y),
                                                               rect_w, rect_h, sampled_points)

    count = 0
    sampled_rectangles = []
    for cur_center in sampled_centers:
        rect = create_rectangle_from_sampled_center(cur_center, map_w=map_w, map_h=map_h, rect_w=rect_w, rect_h=rect_h)
        # check if we can create a rectangle given the sampled center
        if rect:
            cropped_image = crop_rectangle(map_image, rect)
            # count number of white pixels inside the rectangle
            n_of_pixel = 1
            for i in cropped_image.size():
                n_of_pixel *= i
            n_of_white_pixel = torch.count_nonzero(cropped_image)
            white_pixel_rate = torch.div(n_of_white_pixel, n_of_pixel)
            if white_pixel_rate > th:
                sampled_rectangles.append(rect)
                count += 1
        if count == n:
            break
    return sampled_rectangles


def vis_sampled_points(sampled_points, w, h):
    x_coords, y_coords = zip(*sampled_points)

    # Plotting
    plt.figure()
    plt.scatter(x_coords, y_coords, color='blue')
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Uniformly Sampled Points in a Rectangle')
    plt.grid(True)
    plt.show()


def sample_points_in_rectangle(rect_w, rect_h, num_points, r: float = None):
    x_c, y_c = rect_w // 2, rect_h // 2
    if not r:
        r = rect_w / 4
    # TAKE SAMPLE FROM CENTERS TOO, SET R TO 0
    r = 0

    points = []
    while True:
        x = min(rect_w, math.floor(random.uniform(0, rect_w)))
        y = min(rect_h, math.floor(random.uniform(0, rect_h)))
        # check if the given center is enough far away from the source center
        center_diff = ((x_c - x) ** 2 + (y_c - y) ** 2) ** 0.5
        if center_diff >= r:
            points.append((x, y))
        if len(points) == num_points:
            break
    return points


def extract_global_centers_coord_from_points(source_center, rect_w, rect_h, sampled_points):
    x_c, y_c = source_center
    sampled_centers = []
    for point in sampled_points:
        x_t, y_t = point
        x_new = x_c + (x_t - rect_w // 2)
        y_new = y_c + (y_t - rect_h // 2)
        sampled_centers.append((x_new, y_new))
    return sampled_centers


def create_rectangle_from_sampled_center(sampled_center, map_w, map_h, rect_w, rect_h):
    rectangle = None
    x_c, y_c = sampled_center
    cur_center = (x_c, y_c)
    cur_angle = torch.randint(0, 360, (1,)).item()
    cur_rectangle = RectangleCrop(center=cur_center, angle=cur_angle, crop_size=(rect_h, rect_w))
    if is_inside_rectangle(cur_rectangle, num_cols=map_w, num_rows=map_h):
        rectangle = cur_rectangle

    return rectangle


def calculate_diff_between_two_rect(x1: RectangleCrop, x2: RectangleCrop):
    return ((x1.center_x - x2.center_x) ** 2 + (x1.center_y - x2.center_y) ** 2) ** 0.5


def create_unmatched_samples(data: Dict, n, center_difference: float):
    # take all samples
    all_rectangle_samples = []
    for cur_data in data['data']:
        for cur_samples in cur_data['matched_samples']:
            cur_samp_rect_obj = RectangleCrop(center=tuple(cur_samples['center']), angle=cur_samples['angle'],
                                              crop_size=(cur_samples['H'], cur_samples['W']))
            all_rectangle_samples.append(cur_samp_rect_obj)

    new_data_with_unmatched_samples = {'data': []}
    for cur_data in data['data']:
        cur_key_rect_obj: RectangleCrop = RectangleCrop(center=tuple(cur_data['center']),
                                                        angle=cur_data['angle'],
                                                        crop_size=(cur_data['H'], cur_data['W']))
        all_center_diffs: List[float] = list(map(lambda x: calculate_diff_between_two_rect(cur_key_rect_obj, x),
                                                 all_rectangle_samples))

        # take all the rectangles that are far enough from key rectangle
        center_diffs_include_indexes = [idx for idx, diff in enumerate(all_center_diffs) if diff > center_difference]
        filtered_rectangles: List[RectangleCrop] = [all_rectangle_samples[x] for x in center_diffs_include_indexes]

        # randomly select n unmatched samples
        if len(filtered_rectangles) < n:
            return {}
        unmatched_samples: List[RectangleCrop] = random.sample(filtered_rectangles, n)
        unmatched_samples_dicts_list = []
        for cur_unmatched_samp_rect_obj in unmatched_samples:
            cur_unmatch_sample_rect_dict = {
                "center": [cur_unmatched_samp_rect_obj.center_x, cur_unmatched_samp_rect_obj.center_y],
                "angle": cur_unmatched_samp_rect_obj.angle,
                "W": cur_unmatched_samp_rect_obj.crop_w,
                "H": cur_unmatched_samp_rect_obj.crop_h,
                'id': cur_data['id']
            }
            unmatched_samples_dicts_list.append(cur_unmatch_sample_rect_dict)
        cur_data['unmatched_samples'] = unmatched_samples_dicts_list
        new_data_with_unmatched_samples['data'].append(cur_data)

    return new_data_with_unmatched_samples


def augment(map_image_path, json_name, num_sample_rectangles):
    map_image_path = map_image_path
    crop_height = 300
    crop_width = 300
    map_image = load_to_tensor(Path(map_image_path))
    h, w = map_image.shape[1], map_image.shape[2]
    if h <= 3 * crop_height or w <= 3 * crop_width:
        return False

    # sampled_points = sample_points_in_rectangle(crop_width, crop_height, 5)
    # vis_sampled_points(sampled_points, crop_width, crop_width)

    patches = calculate_patches(image_height=h, image_width=w, patch_height=crop_height, patch_width=crop_width,
                                overlap_width_ratio=0.0, overlap_height_ratio=0.0)

    patches = filter_patches(map_image, patches, p_h=crop_height, p_w=crop_width, threshold=0.4)

    num_sample_rectangles = num_sample_rectangles
    rectangle_data = {"data": []}
    key_rect_id = 0
    for box in patches:
        x_min, y_min = box[0], box[1]
        x_max, y_max = box[2], box[3]
        x_c = min(x_max, x_min) + abs(x_max - x_min) // 2
        y_c = min(y_max, y_min) + abs(y_max - y_min) // 2
        cur_center = (x_c, y_c)
        cur_angle = torch.randint(0, 360, (1,)).item()
        cur_rectangle = RectangleCrop(center=cur_center, angle=cur_angle, crop_size=(crop_height, crop_width))
        if not is_inside_rectangle(cur_rectangle, num_cols=w, num_rows=h):
            continue
        sampled_rects = sample_n_rectangles(map_image=map_image, source_rect=cur_rectangle,
                                            rect_h=crop_height, rect_w=crop_width, n=num_sample_rectangles)

        cur_key_rect_dict = {
            "center": [cur_rectangle.center_x, cur_rectangle.center_y],
            "angle": cur_rectangle.angle,
            "W": cur_rectangle.crop_w,
            "H": cur_rectangle.crop_h,
            "matched_samples": [],
            "id": key_rect_id
        }

        for cur_samp_rect in sampled_rects:
            cur_sample_rect_dict = {
                "center": [cur_samp_rect.center_x, cur_samp_rect.center_y],
                "angle": cur_samp_rect.angle,
                "W": cur_samp_rect.crop_w,
                "H": cur_samp_rect.crop_h,
                'id': key_rect_id
            }
            cur_key_rect_dict['matched_samples'].append(cur_sample_rect_dict)

        rectangle_data['data'].append(cur_key_rect_dict)
        key_rect_id += 1

    if not rectangle_data['data']:
        return False
    
    center_difference = 4 * crop_height // 2 * 2 ** 0.5  # discard each 3 rectangle outside the corresponding rect.
    rectangle_data = create_unmatched_samples(rectangle_data, num_sample_rectangles,
                                              center_difference=center_difference)
    if rectangle_data:
        save_to_json(rectangle_data, f"dataset/data_meta/{json_name}.json")
        # visualize_rect_dict_on_map(map_image, rectangle_data)
        return True
    else:
        return False


if __name__ == "__main__":
    maps_path = Path("/home/melihgorgulu/cartken/Cartken_Scan_Matching/data/dataset/maps")
    for f_name in maps_path.iterdir():
        if ".DS_Store" in str(f_name):
            continue
        map_name = f_name.name
        map_image_path = f_name / "map.png"
        ack = augment(map_image_path, str(map_name), num_sample_rectangles=10)
        if not ack:
            logger.warning(f"{map_name} size is not sufficient to create crops with specified size.")
        else:
            logger.info(f"{map_name} is successfully augmented")

    print("DONE!!")

