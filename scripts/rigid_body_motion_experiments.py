import torch
from PIL import Image
import torchvision.transforms as transforms
import math
import torchvision
from typing import Tuple
import torchvision.transforms.functional


class ThresholdCrop:
    def __init__(self, n: int = 50, threshold: float = 0.5, size_h: int = 320, size_w: int = 320):
        self._n = n
        self._threshold = threshold
        self._crop_h = math.ceil(size_h * math.sqrt(2))
        self._crop_w = math.ceil(size_w * math.sqrt(2))
        self._size_h = size_h
        self._size_w = size_w

    def __call__(
            self, image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input: BS x C x H x W, assuming C == 1
        # Output: Tuple[BS x N x H' x W', BS x N x 2, BS x N x 2]
        # of cropped image, translation ground truth and rotation ground truth
        bs, c, h, w = image.shape
        range_h, range_w = h - self._crop_h, w - self._crop_h
        assert range_h * range_w >= self._n, \
            f'Input is too small to give {self._n} distinct random crops.'
        # Random permute all the positions (top-left corner) possible to crop
        choices = torch.randperm(int(range_h * range_w))
        # Uncomment the line below to make the positions deterministic
        # choices = range(int(range_h * range_w))
        output_img = torch.zeros((bs, self._n, self._size_h, self._size_w))
        tr_gt = torch.zeros((bs, self._n, 2))
        # Store rotation as x and y coords on unit circle
        rot_gt = torch.zeros((bs, self._n, 2))
        for b in range(bs):
            n = 0
            while n < self._n:
                for c in choices:
                    # Randomly pick
                    deg = torch.randint(0, 360, (1,))
                    rad = deg / 180 * torch.pi
                    # Use torch.div instead of `//` to depress warnings
                    u_tl, v_tl = torch.div(c, range_w, rounding_mode='floor'), c % range_w
                    u_c, v_c = u_tl + self._crop_h // 2, v_tl + self._crop_w // 2

                    # Crop from the original input and rotate it
                    cropped = image[b][..., u_tl:u_tl + self._crop_h, v_tl:v_tl + self._crop_w]
                    rotated = torchvision.transforms.functional.rotate(cropped, angle=float(deg))
                    cropped = torchvision.transforms.functional.center_crop(
                        rotated, [self._size_h, self._size_w])
                    # if cropped.mean() >= self._threshold * avg_intensity:
                    n_pixels_cropped = 1
                    for s in cropped.shape:
                        n_pixels_cropped *= s
                    if cropped.count_nonzero() > 0.5 * n_pixels_cropped:
                        # Add it if threshold is exceeded
                        output_img[b, n, ...] = cropped
                        # Also return the gt of translation and rotation
                        tr_gt[b, n] = torch.Tensor([u_c, v_c])
                        rot_gt[b, n] = torch.Tensor([
                            torch.cos(torch.Tensor(rad)),
                            torch.sin(torch.Tensor(rad))
                        ])
                        n += 1
                        if n >= self._n:
                            break

        return output_img, tr_gt, rot_gt


example_data_path = r"/Users/melihgorgulu/Desktop/Projects/Cartken/Cartken_Scan_Matching/data/dataset/maps/aeon-toki/map.png"

image = Image.open(example_data_path)

transform = transforms.Compose([
    transforms.PILToTensor()
])

img_tensor = transform(image)
# add batch dimension
img_tensor = torch.unsqueeze(img_tensor, 0)


th_crop = ThresholdCrop(n=10)

out_img, translation, rotation = th_crop(img_tensor)
print('stop')
