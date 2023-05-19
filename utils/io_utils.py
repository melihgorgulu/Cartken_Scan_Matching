import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
import json


def convert_pil_to_tensor(img: Image, mode='gray') -> torch.Tensor:
    tensor_convertor = transforms.ToTensor()

    out = tensor_convertor(img)
    if out.shape[0] == 3 and mode == 'gray':
        out = out[0, :, :]
        out = torch.unsqueeze(out, dim=0)
        return out
    return out


def convert_tensor_to_pil(img: torch.Tensor) -> Image:
    pil_convertor = transforms.ToPILImage()
    out = pil_convertor(img)
    return out


def show_tensor_image(img: torch.Tensor):
    pil_image = convert_tensor_to_pil(img)
    pil_image.show()


def save_tensor_as_image(img: torch.Tensor, save_path: Path):
    save_image(img, str(save_path))


def save_to_json(dictionary: dict, json_path: str, indent=2):
    with open(json_path, 'w') as fp:
        json.dump(dictionary, fp, indent=indent, sort_keys=True)


def read_json(json_path: Path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_to_tensor(image_path: Path) -> torch.Tensor:
    # Load the image using PIL
    image = Image.open(image_path)
    # Convert the image to a PyTorch tensor
    tensor = convert_pil_to_tensor(image)
    return tensor
