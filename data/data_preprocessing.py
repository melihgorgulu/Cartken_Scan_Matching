import torch
from typing import Tuple
from data_utils import *
import logging
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 933120000


logger = logging.getLogger()


class RectangleCrop:
    def __init__(self, center: Tuple[int, int], angle: float, crop_size: Tuple[int, int] = (320, 320)):
        self.center_x = center[0]
        self.center_y = center[1]
        self.crop_h, self.crop_w = crop_size
        self.angle = angle
        # generator with same seed to get same results.
        # g_cpu = torch.Generator()
        # g_cpu.manual_seed(random_seed)
        # use torch.randint (uniform sampling)
        # self.angle = torch.randint(low=0, high=360, size=(1,), generator=random_generator)

    def __str__(self):
        out = f"Rectangle: ({self.crop_h}x{self.crop_w})\n"
        out += f"Center x: {self.center_x}, Center y: {self.center_y}\n"
        out += f"Angle: {self.angle}\n"
        return out


class RandomRotatedCrop:
    def __init__(self, show_generated_rectangle: bool = False):
        self.show = show_generated_rectangle

    def __call__(self, map_image: torch.Tensor, rectangle: RectangleCrop):
        map_rows = map_image.shape[0]
        map_cols = map_image.shape[1]
        if is_inside_rectangle(rectangle, map_cols, map_rows):
            if self.show:
                visualize_crop_on_map(map_image, rectangle, f_name="example_random_crop")
            rotated_angle = rectangle.angle

            rect_bbx_upright = rect_bbx(rect=rectangle)
            rect_in_obj = RectangleCrop(center=rect_bbx_upright[0], angle=rect_bbx_upright[2],
                                        crop_size=rect_bbx_upright[1])
            rect_bbx_upright_image = crop_rectangle(image=map_image, rect=rect_in_obj)

            rotated_rect_bbx_upright_image = image_rotate_without_crop(mat=rect_bbx_upright_image, angle=rotated_angle)

            rect_width = rectangle.crop_w
            rect_height = rectangle.crop_h

            crop_center = (rotated_rect_bbx_upright_image.shape[1] // 2, rotated_rect_bbx_upright_image.shape[0] // 2)
            out = rotated_rect_bbx_upright_image[
                  crop_center[1] - rect_height // 2: crop_center[1] + (rect_height - rect_height // 2),
                  crop_center[0] - rect_width // 2: crop_center[0] + (rect_width - rect_width // 2)]
            out = torch.from_numpy(out)
            return out

        else:
            logger.error("Given crop size is not suitable for the given map !!")
            raise RuntimeError