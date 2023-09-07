
from utils.io_utils import *
from data.data_preprocessing import *

def test_augmentation():
    meta_path = "/home/melihgorgulu/cartken/Cartken_Scan_Matching/data/dataset/data_meta/aeon-toki.json"
    meta = read_json(meta_path)
    breakpoint()
    rect_center = meta['data'][0]['center']
    rect_angle = meta['data'][0]['angle']
    rect_source = RectangleCrop(center=rect_center, angle=rect_angle, crop_size=(300,300))
    rect_center = meta['data'][0]['matched_samples'][0]['center']
    rect_angle = meta['data'][0]['matched_samples'][0]['angle']
    rect_target = RectangleCrop(center=rect_center, angle=rect_angle, crop_size=(300,300))
    gt_tx = rect_source.center_x - rect_target.center_x
    gt_ty = rect_source.center_y - rect_target.center_y
    angle_diff = rect_source.angle - rect_target.angle
    if angle_diff < 0:
        angle_diff = 360 + angle_diff
    
    


if __name__ == "__main__":
    test_augmentation()
