import os
from data.data_model import ScanMatchingDataSet
from data.transforms import Standardize, ResNet50_Transforms
from data.data_model import DatasetFromSubset
from torch.utils.data import DataLoader
from utils.config import get_train_config, get_data_config, get_stats_config
import torch
import torch.optim as optim
from data.data_utils import random_split
import numpy as np
from typing import Tuple

from model.networks import BasicSMNetwork, SmNetwithResNetBackBone, SmNetwithResNetBackBone_Small
from training.losses import CombinedLoss
from training.trainer import SMNetTrainer

from pathlib import Path
from typing import Dict, List
from utils.io_utils import read_json, save_to_json
from scripts.calculate_train_statistics import calculate_all_stats

from torchvision.transforms import Compose


def _to_device(tensors: Tuple[torch.Tensor], device) -> List[torch.Tensor]:
    tensors = list(tensors)
    tensors = list(map(lambda x: x.to(device), tensors))
    return tensors

def get_gt_and_pred(model_path:str, save=False, use_saved=False):
    
    if use_saved:
        print("LOADING NUMPY ARRAY")
        with open('gt_match.npy', 'rb') as f:
            all_match_gt = np.load(f, allow_pickle=True)
        with open('gt_trans.npy', 'rb') as f:
            all_transform_gt = np.load(f, allow_pickle=True)
            
        with open('pred_match.npy', 'rb') as f:
            all_match_preds = np.load(f, allow_pickle=True)

        with open('pred_trans.npy', 'rb') as f:
            all_transform_preds = np.load(f, allow_pickle=True)
        print("LOAD COMPLETED")
        return all_match_gt, all_transform_gt, all_match_preds, all_transform_preds
            
            
    train_config = get_train_config()
    stats_config = get_stats_config()
    tx_max, ty_max = stats_config["tx_stats"]["max"], stats_config["ty_stats"]["max"]
    # test dataset loader
    full_dataset = ScanMatchingDataSet()
    train_size, val_size, test_size = train_config["TRAIN_SIZE"], train_config["VAL_SIZE"], train_config["TEST_SIZE"]
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
    print(F"Test Dataset size: {len(test_dataset)}")
    transform_train = Compose([ResNet50_Transforms()]) # use default std mean
    test_dataset = DatasetFromSubset(test_dataset, transform=transform_train)
    batch_size = 10
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # load the model
    model = SmNetwithResNetBackBone_Small()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to("cuda")
    
    all_match_preds = np.zeros(shape=(len(test_dataset),1), dtype=np.float32)
    all_transform_preds = np.zeros(shape=(len(test_dataset), 4), dtype=np.float32)
    
    all_match_gt = np.zeros(shape=(len(test_dataset),1), dtype=np.float32)
    all_transform_gt = np.zeros(shape=(len(test_dataset), 4), dtype=np.float32)
    for i, test_data in enumerate(test_loader):
        print(f"Index:{i}/{len(test_loader)}")
        test_data = _to_device(test_data, device='cuda')
        source_patches, target_patches, gt_match, gt_transform = test_data
        match_logits, transform_pred = model(source_patches, target_patches)

        match_pred = torch.sigmoid(match_logits).cpu().detach().numpy()
        match_pred = np.where(match_pred<0.5,0.0,1.0)
        all_match_preds[i*batch_size:i*batch_size+batch_size,...] = match_pred
        
        # consider scaling
        transform_pred[..., 2] = transform_pred[..., 2] * tx_max
        transform_pred[..., 3] = transform_pred[..., 3] * ty_max
        
        transform_pred = transform_pred.cpu().detach().numpy()
        
        all_transform_preds[i*batch_size:i*batch_size+batch_size,...] = transform_pred
        
        # ground truths
        all_match_gt[i*batch_size:i*batch_size+batch_size,...] = gt_match.cpu().detach().numpy()
        all_transform_gt[i*batch_size:i*batch_size+batch_size,...] = gt_transform.cpu().detach().numpy()
        
    # filter out non-matched pairs from transform ground truths
    
    match_mask_indices = [idx for idx, val in enumerate(all_match_gt[:, 0].tolist()) if val == 1.0]  # take the batch
    all_transform_preds = all_transform_preds[match_mask_indices,...]
    all_transform_gt = all_transform_gt[match_mask_indices,...]
    
    if save:
        print("SAVING NUMPY ARRAY")
        with open('gt_match.npy', 'wb') as f:
            np.save(f,all_match_gt)
        with open('gt_trans.npy', 'wb') as f:
            np.save(f,all_transform_gt)
            
        with open('pred_match.npy', 'wb') as f:
            np.save(f,all_match_preds)

        with open('pred_trans.npy', 'wb') as f:
            np.save(f,all_transform_preds)
    return all_match_gt, all_transform_gt, all_match_preds, all_transform_preds


def plot_roc(gt, pred):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(gt, pred)

    # Calculate the AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig("roc_curve.png")


    
def eval_match(gt, pred):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(gt, pred)
    acc = accuracy_score(gt, pred)
    precision = precision_score(gt, pred)
    recall = recall_score(gt, pred)
    f1 = f1_score(gt, pred)
    # roc curve
    plot_roc(gt, pred)

    
def calc_mse(gt,pred):
    return np.mean((gt - pred) ** 2)

def calc_rmse(gt,pred):
    return calc_mae(gt,pred)**(0.5)

def calc_mae(gt, pred):
    return np.mean(np.abs(gt - pred))

def calc_r2(gt, pred):
    from sklearn.metrics import r2_score
    return r2_score(gt, pred)


def eval_transform(gt, pred):
    breakpoint()
    gt_tx, pred_tx = gt[..., 2], pred[..., 2]
    gt_ty, pred_ty = gt[..., 3], pred[..., 3] 
    
    gt_sin, pred_sin = gt[..., 1], pred[..., 1]
    gt_cos, pred_cos = gt[..., 0], pred[..., 0] 
    
    mse_tx = calc_mse(gt_tx,pred_tx)
    mse_ty = calc_mse(gt_ty,pred_ty)
    mse_sin = calc_mse(gt_sin,pred_sin)
    mse_cos = calc_mse(gt_cos,pred_cos)
    
    mae_tx = calc_mae(gt_tx,pred_tx)
    mae_ty = calc_mae(gt_ty,pred_ty)
    mae_sin = calc_mae(gt_sin,pred_sin)
    mae_cos = calc_mae(gt_cos,pred_cos)
    
    r2_tx = calc_r2(gt_tx,pred_tx)
    r2_ty = calc_r2(gt_ty,pred_ty)
    r2_sin = calc_r2(gt_sin,pred_sin)
    r2_cos = calc_r2(gt_cos,pred_cos)
    print("heyy")
    print("stop")
    


def evaluate():
    model_path = "/home/melihgorgulu/cartken/Cartken_Scan_Matching/training/trained_models/22_09_2023_gridsearch_5_lr0.00_wd_0.00_small.pt"
    all_match_gt, all_transform_gt, all_match_preds, all_transform_preds = get_gt_and_pred(model_path)
    eval_match(all_match_gt, all_match_preds)
    eval_transform(all_transform_gt, all_transform_preds)
    
if __name__ == "__main__":
    evaluate()