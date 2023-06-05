import os
from torch import Tensor
import math
from utils.io_utils import *
from data_utils import calculate_patches
from torchvision import transforms
from typing import Tuple, List
from torchvision.transforms.functional import affine
import torch.nn.functional as nnf

Image.MAX_IMAGE_PIXELS = None


class PatchGenerator:
    def __init__(self, save_dir: Path, patch_height=300, patch_width=300, overlap_height_ratio: float = 0.2,
                 overlap_width_ratio: float = 0.2):
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.save_dir = save_dir
        if not save_dir.exists():
            save_dir.mkdir()

    def filter_patches(self, image: Tensor, patches, threshold: float = 0.5):
        index_to_exclude = []
        for cur_idx, cur_patch_coord in enumerate(patches):
            x1, y1, x2, y2 = cur_patch_coord[0], cur_patch_coord[1], cur_patch_coord[2], cur_patch_coord[3]
            cur_patch = image[:, y1:y2, x1:x2]
            if cur_patch.shape[1] != self.patch_height or cur_patch.shape[2] != self.patch_width:
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

    def __call__(self, img_path: Path):
        with Image.open(img_path) as cur_img:
            cur_img_tensor = convert_pil_to_tensor(cur_img)
            c, h, w = cur_img_tensor.shape
            patches = calculate_patches(image_height=h, image_width=w, patch_width=self.patch_width,
                                        patch_height=self.patch_height, overlap_height_ratio=self.overlap_height_ratio,
                                        overlap_width_ratio=self.overlap_width_ratio)

            filtered_patches = self.filter_patches(cur_img_tensor, patches, threshold=0.4)
            # visualise_patches_on_img(str(img_path), filtered_patches)
            for cur_idx, cur_patch_coord in enumerate(filtered_patches):
                x1, y1, x2, y2 = cur_patch_coord[0], cur_patch_coord[1], cur_patch_coord[2], cur_patch_coord[3]
                cur_patch = cur_img_tensor[:, y1:y2, x1:x2]
                f_name = f"patch_{img_path.parent.name}_{cur_idx}.png"
                save_path = self.save_dir / f_name
                save_image(cur_patch, save_path)


def create_patch_dataset(root_dir: Path):
    patch_generator = PatchGenerator(Path(os.getenv("PATCH_SAVE_DIR")))
    for cur_dir in root_dir.iterdir():
        if ".DS_Store" in str(cur_dir):
            continue
        for cur_img_path in cur_dir.iterdir():
            if 'txt' in cur_img_path.name:
                continue
            patch_generator(cur_img_path)


class RandomAffineTransform:
    def __init__(self, deg_max=180, deg_min=-180, h_t=0.2, v_t=0.2):
        self.deg_min = deg_min
        self.deg_max = deg_max
        self.v_t = v_t
        self.h_t = h_t
        self.degree = None
        self.translations = None

    def __call__(self, img: Tensor) -> Tuple[torch.Tensor, float, List[int]]:
        c, h, w = img.shape
        degree, translations, scale, shear = transforms.RandomAffine.get_params(degrees=[self.deg_min, self.deg_max],
                                                                                translate=[self.h_t, self.v_t],
                                                                                scale_ranges=None, shears=None,
                                                                                img_size=[h, w])
        self.degree = degree
        self.translations = translations

        # apply rigid body transformation
        transformed_image = affine(img, angle=self.degree, translate=list(self.translations), scale=scale,
                                   shear=list(shear))
        # degree to radian
        radians = math.radians(self.degree)
        return transformed_image, radians, list(self.translations)

    def show_transformed_image(self, img: Tensor):
        c, h, w = img.shape
        angle, translations, scale, shear = transforms.RandomAffine.get_params(degrees=[self.deg_min, self.deg_max],
                                                                               translate=[self.h_t, self.v_t],
                                                                               scale_ranges=None, shears=None,
                                                                               img_size=[c, h, w])
        angle = float(round(angle))
        transformed_image = affine(img, angle=angle, translate=list(translations), scale=scale,
                                   shear=list(shear))
        show_tensor_image(img)
        show_tensor_image(transformed_image)


"""
def test_random_affine():
    im = torch.ones(size=(1, 512, 512))
    for i in range(100):
        random_affine_transform = RandomAffineTransform(deg_max=180, deg_min=-180, h_t=0.2, v_t=0.2)
        _, radians, t = random_affine_transform(im)
        print(radians, t, sep=" ")
"""


def create_transformation_dataset():
    random_affine_transform = RandomAffineTransform(deg_max=180, deg_min=-180, h_t=0.2, v_t=0.2)
    patches_dir = Path(os.getenv("PATCH_SAVE_DIR"))
    gt_dict = {'data': []}
    for index, cur_patch_path in enumerate(patches_dir.iterdir()):
        new_name = cur_patch_path.parent / f"{index}{cur_patch_path.suffix}"
        trans_image_save_name = new_name.parent.parent / f"transformed_patches/{index}_trans{cur_patch_path.suffix}"

        # rename patch by iteration index
        cur_patch_path.rename(new_name)

        with Image.open(new_name) as cur_img:
            cur_img = convert_pil_to_tensor(cur_img)
            transformed_img, radian, translation = random_affine_transform(cur_img)
            gt_rot = [math.cos(radian), math.sin(radian)]
            gt_trans = translation
            cur_lbl_dict = {"org_patch_name": new_name.name,
                            "translated_patch_name": trans_image_save_name.name,
                            "data_idx": index,
                            "gt_rot": gt_rot,
                            "gt_trans": gt_trans}
            gt_dict['data'].append(cur_lbl_dict)
            save_tensor_as_image(transformed_img, save_path=trans_image_save_name)
            print(index)
    save_to_json(dictionary=gt_dict, json_path=os.getenv("LABEL_SAVE_PATH"))


def main():
    # create_patch_dataset(root_dir=Path(os.getenv("ROOT_DIR")))
    create_transformation_dataset()
    print('Done!')


if __name__ == "__main__":
    main()
