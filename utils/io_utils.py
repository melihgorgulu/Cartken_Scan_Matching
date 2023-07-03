import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
import json
import matplotlib.pyplot as plt
from typing import List
import math
from torchvision.transforms.functional import affine


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


def save_loss_graph(save_path: Path, train_loss, val_loss, titles: List[str], labels: List[str]):
    plt.figure()
    plt.subplot(1, 2, 1)  # Create a subplot for the first plot (training loss)
    plt.plot(train_loss, 'r-', label=labels[0])  # 'r-' denotes red color and line style
    plt.title(titles[0])  # Set the title for the training loss plot
    plt.xlabel('Epochs')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.legend()  # Show the legend

    # Plotting the validation loss
    plt.subplot(1, 2, 2)  # Create a subplot for the second plot (validation loss)
    plt.plot(val_loss, 'b-', label=labels[1])  # 'b-' denotes blue color and line style
    plt.title(titles[1])  # Set the title for the validation loss plot
    plt.xlabel('Epochs')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.legend()  # Show the legend

    # Display the plot
    plt.tight_layout()  # Adjust the layout to avoid overlapping labels
    plt.savefig(str(save_path))


def revert_image_transform(img: torch.Tensor, transformation: torch.Tensor):
    """
    Given transformed image ant applied transformation, returns the original image
    :param img: transformed image
    :param transformation: used transformation matrix
    :return: original image
    """
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
    translate_inv = [-tx, -ty]
    # first translate back
    reverted_img = affine(img, angle=0, translate=translate_inv, scale=1.0, shear=[0.0, 0.0])
    # then rotate
    reverted_img = affine(reverted_img, angle=deg_inv, translate=[0, 0], scale=1.0, shear=[0.0, 0.0])
    return reverted_img
