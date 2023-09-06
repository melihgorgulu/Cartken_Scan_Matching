from data.data_model import ScanMatchingDataSet
from utils.io_utils import show_tensor_image, convert_tensor_to_pil
from pathlib import Path
import random
import torch
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt


def test_dataset():
    sm = ScanMatchingDataSet()
    img, trans_img, lbl_match, lbl_transform = sm[random.randint(0, len(sm))]
    show_tensor_image(img)
    show_tensor_image(trans_img)
    print(lbl_transform)
    print(lbl_match)
    print(len(sm))


def test_dataloader():
    sm = ScanMatchingDataSet()
    train_size = 0.7
    test_size = 0.1
    val_size = 0.2

    gen = torch.Generator().manual_seed(42)
    train_data, val_data, test_data = random_split(sm, [train_size, val_size, test_size], generator=gen)
    train_loader = DataLoader(train_data)
    val_loader = DataLoader(val_data)
    test_loader = DataLoader(test_data)
    print(len(train_loader), len(val_loader), len(test_loader))
    for idx, item in enumerate(train_loader):
        img, trans_im, lbl_transform, lbl_match = item
        print(img.shape, trans_im.shape, lbl_transform, lbl_match)
        if idx == 3:
            break

def vis_dataloader():
    sm = ScanMatchingDataSet()
    k = 15
    for i in range(k):
        img, trans_img, lbl_match, lbl_transform = sm[random.randint(0, len(sm))]
        img = convert_tensor_to_pil(img)
        trans_img = convert_tensor_to_pil(trans_img)
        # Set the titles for each image
        if lbl_match[0].item()==1:
            title1 = "Matched Pairs"
        else:
            title1 = "Un-Matched Pairs"
        
        cosx,sinx,tx,ty = lbl_transform
        cosx = cosx.item()
        sinx = sinx.item()
        tx = tx.item()
        ty = ty.item()
        
        title2 = f"Cost: {cosx}, Sint: {sinx}, Tx: {tx}, Ty: {ty}"

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Display the first image in the left subplot
        ax1.imshow(img, cmap="gray")
        ax1.set_title(title1)
        ax1.axis('off')  # Turn off axis labels

        # Display the second image in the right subplot
        ax2.imshow(trans_img, cmap="gray")
        ax2.set_title(title2)
        ax2.axis('off')  # Turn off axis labels

        # Save the figure as an image file (e.g., PNG)
        plt.savefig(f"dataset_preview/data_{i}.png", bbox_inches='tight')  # Specify the desired file format
        print("Done")
   

    


if __name__ == "__main__":
    vis_dataloader()
    #test_dataset()
    #test_dataloader()
