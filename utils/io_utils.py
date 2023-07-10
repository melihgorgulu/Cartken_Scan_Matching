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

def save_prediction_results(org_img: torch.Tensor, trans_img: torch.Tensor,
                            gt_match: torch.Tensor,
                            gt_trans: torch.Tensor, 
                            prediction_affine: torch.Tensor, 
                            prediction_match: torch.Tensor, 
                            experiment_name: str, epoch: int):
    """_summary_
        Save model prediction results for given epoch
    """
    save_dir = Path("experiments") / experiment_name / "predictions_vis" / f"epoch_{epoch}"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    batch_size = prediction_affine.shape[0]
    for i in range(batch_size):
        cur_gt_match = gt_match[i, ...] 
        gt_cost, gt_sint, gt_tx, gt_ty = gt_trans[i, ...]
        cost, sint, tx, ty = prediction_affine[i, ...]
        match_score = prediction_match[i, ...]
        # we have logits for match score, apply sigmoid to it
        match_score = torch.sigmoid(match_score)
        predicted_image = revert_image_transform(trans_img[i,...], prediction_affine[i, ...])
        # convert tensor to pil images for saving
        cur_org_img = convert_tensor_to_pil(org_img[i,...])
        cur_trans_img = convert_tensor_to_pil(trans_img[i,...])
        cur_pred_img = convert_tensor_to_pil(predicted_image)
        """
        save_path_org = save_dir / f"epoch_{epoch}_data_{i}_original.png"
        cur_org_img.save(save_path_org, "PNG")
        
        save_path_trans = save_dir / f"epoch_{epoch}_data_{i}_trans.png"
        cur_trans_img.save(save_path_trans, "PNG")

        save_path_pred = save_dir / f"epoch_{epoch}_data_{i}_pred_.png"
        cur_pred_img.save(save_path_pred, "PNG")
        
        """
        # Create a figure with three subplots arranged side by side
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(cur_org_img)
        axs[0].axis('off')  # Disable axes
        axs[0].set_title("Original Image")
        # Display the second image on the second subplot
        axs[1].imshow(cur_trans_img)
        axs[1].set_title("Trans Image")
        axs[1].axis('off')  # Disable axes

        # Display the third image on the third subplot
        axs[2].imshow(cur_pred_img)
        axs[2].set_title("Pred Image")
        axs[2].axis('off')  # Disable axes

        main_title = f" Gt Match: {cur_gt_match.item():.2f}, Gt cost: {gt_cost.item():.2f}, Gt sint:{gt_sint.item():.2f}, Gt tx: {gt_tx.item():.2f}, Gt ty: {gt_ty.item():.2f} \n Pred Match: {match_score.item():.2f}, Pred cost: {cost.item():.2f}, Pred sint: {sint.item():.2f}, Pred tx: {tx.item():.2f}, Pred ty: {ty.item():.2f}"
        fig.suptitle(main_title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        save_path = save_dir / f"epoch_{epoch}_data_{i}.png"
        plt.savefig(str(save_path))