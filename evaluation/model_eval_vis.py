
from utils.io_utils import *
import random
import numpy as np
import cv2
from utils.config import get_train_config, get_data_config, get_stats_config
import torch
from data.transforms import ResNet50_Transforms
from data.data_preprocessing import RectangleCrop, RandomRotatedCrop
from model.networks import SmNetCorrResNetBackBone, SmNetwithResNetBackBone_Small

stats_config=get_stats_config()
tx_max, ty_max = stats_config["tx_stats"]["max"], stats_config["ty_stats"]["max"]



def apply_transform(map_image, gt_angle_diff, source_rect: RectangleCrop, pred_transformation):

    cost, sint, tx, ty = pred_transformation
    deg = abs(math.degrees(math.acos(cost)))

    predicted_angle = source_rect.angle + deg
    if predicted_angle>360:
        predicted_angle = predicted_angle - 360
    predicted_angle = round(predicted_angle)
    predicted_center_x = round(source_obj.center_x + tx.item())
    predicted_center_y = round(source_obj.center_y + ty.item())
    predicted_rectangle = RectangleCrop(center=(predicted_center_x, predicted_center_y), angle=predicted_angle,
                                        crop_size=(source_rect.crop_h,source_rect.crop_w))
    
    cropper = RandomRotatedCrop()
    pred_img = cropper(map_image, predicted_rectangle)
    
    return pred_img
        
    

def visualize_crops_on_map(image: torch.Tensor, rectangles):
    image = image.permute(1, 2, 0) # take channel dim to the last
    image = image.detach().cpu().numpy()  # torch tensor to numpy array
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Create a window to display the image

    current_index = 0

    # Loop until all bounding boxes have been drawn
    while current_index < len(rectangles):
        # Draw the current bounding box on the image
        rectangle = rectangles[current_index]
        rect_center = rectangle["center"]
        rect_wh = (rectangle["W"], rectangle["H"])
        angle = rectangle["angle"]
        rect_in = (rect_center, rect_wh, angle)
        box = cv2.boxPoints(rect_in).astype(np.int64)
        if current_index==2:
            cv2.drawContours(image, [box], 0, (0, 0, 255), 10)
            cv2.arrowedLine(image, rect_center, ((box[1][0] + box[2][0]) // 2, (box[1][1] + box[2][1]) // 2),
                            (0, 0, 255),
                            3, tipLength=0.2)
        else:         
            cv2.drawContours(image, [box], 0, (255, 0, 0), 10)
            cv2.arrowedLine(image, rect_center, ((box[1][0] + box[2][0]) // 2, (box[1][1] + box[2][1]) // 2),
                            (255, 0, 0),
                            3, tipLength=0.2)

        current_index += 1
        

    cv2.imwrite("test.jpg", image)



def vis_prediction(map_image, source_rect, org_img: torch.Tensor, trans_img: torch.Tensor,
                   gt_match: torch.Tensor, gt_trans: torch.Tensor, 
                   prediction_affine: torch.Tensor, prediction_match: torch.Tensor, gt_angle_diff, idx):
    gt_cost, gt_sint, gt_tx, gt_ty = gt_trans
    pred_cost, pred_sint, pred_tx, pred_ty = prediction_affine[0]
    match_score = prediction_match[0]
    predicted_image = apply_transform(map_image, gt_angle_diff,source_rect, prediction_affine[0])
    
    org_img = convert_tensor_to_pil(org_img)
    trans_img = convert_tensor_to_pil(trans_img)
    predicted_image = convert_tensor_to_pil(predicted_image)
    
    # Create a figure with three subplots arranged side by side
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(org_img)
    axs[0].axis('off')  # Disable axes
    axs[0].set_title("Original Image")
    # Display the second image on the second subplot
    axs[1].imshow(trans_img)
    axs[1].set_title("Trans Image")
    axs[1].axis('off')  # Disable axes

    # Display the third image on the third subplot
    axs[2].imshow(predicted_image)
    axs[2].set_title("Pred Image")
    axs[2].axis('off')  # Disable axes

    main_title = f" Gt Match: {gt_match:.2f}, Gt cost: {gt_cost.item():.2f}, Gt sint:{gt_sint.item():.2f}, Gt tx: {gt_tx.item():.2f}, Gt ty: {gt_ty.item():.2f} \n Pred Match: {match_score.item():.2f}, Pred cost: {pred_cost.item():.2f}, Pred sint: {pred_sint.item():.2f}, Pred tx: {pred_tx.item():.2f}, Pred ty: {pred_ty.item():.2f}"
    fig.suptitle(main_title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = f"pred_{i}.png"
    plt.savefig(save_path)
    print('prediction is saved')

map_path = "/home/melihgorgulu/cartken/Cartken_Scan_Matching/data/dataset/maps/aeon-toki/map.png"
map_meta_path = "/home/melihgorgulu/cartken/Cartken_Scan_Matching/data/dataset/data_meta/aeon-toki.json"
model_path = "/home/melihgorgulu/cartken/Cartken_Scan_Matching/training/trained_models/22_09_2023_gridsearch_5_lr0.00_wd_0.00_small.pt"


map_image = load_to_tensor(Path(map_path))

map_meta = read_json(map_meta_path)

n = 5

for i in range(n):
    if len(map_image.shape) !=3:
        map_image = torch.unsqueeze(map_image,0)
    source_random_idx = random.randint(0, len(map_meta['data'])-1)

    source_patch = map_meta['data'][source_random_idx]

    target_matched_random_idx = random.randint(0, len(source_patch['matched_samples'])-1)
    target_umatched_random_idx = random.randint(0, len(source_patch['unmatched_samples'])-1)

    target_matched_patch = source_patch['matched_samples'][target_matched_random_idx]

    target_umatched_patch = source_patch['unmatched_samples'][target_umatched_random_idx]


    crop_height, crop_width  = 300, 300
    h, w = map_image.shape[1], map_image.shape[2]

    #visualize_crops_on_map(map_image, [source_patch, target_matched_patch, target_umatched_patch])


    # remove 3d dim
    map_image = map_image[0,...]
    cropper = RandomRotatedCrop()
    source_obj = RectangleCrop(center=tuple(source_patch['center']),
            angle=source_patch['angle'],
            crop_size=(source_patch['H'], source_patch['W']))
    t_match_obj = RectangleCrop(center=tuple(target_matched_patch['center']),
            angle=target_matched_patch['angle'],
            crop_size=(target_matched_patch['H'], target_matched_patch['W']))
    t_unmatch_obj = RectangleCrop(center=tuple(target_umatched_patch['center']), 
                    angle=target_umatched_patch['angle'], 
                    crop_size=(target_umatched_patch['H'], target_umatched_patch['W']))

    transform = ResNet50_Transforms()

    # add bacth dim
    s_im = torch.unsqueeze(transform(torch.unsqueeze(cropper(map_image, source_obj),0)),0).to("cuda")
    t_match_im = torch.unsqueeze(transform(torch.unsqueeze(cropper(map_image, t_match_obj),0)),0).to("cuda")
    t_unmatch_im = torch.unsqueeze(transform(torch.unsqueeze(cropper(map_image, t_unmatch_obj),0)),0).to("cuda")


    model = SmNetwithResNetBackBone_Small()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to("cuda")
    # prediction with macted pair
    # matched pair
    pred_match, pred_trans = model(s_im, t_match_im)
    pred_match = torch.sigmoid(pred_match)
    pred_trans[..., 2] = pred_trans[..., 2] * tx_max
    pred_trans[..., 3] = pred_trans[..., 3] * ty_max


    # unmatched pair
    pred_match_2, pred_trans_2 = model(s_im, t_unmatch_im)
    pred_match_2 = torch.sigmoid(pred_match)

    pred_trans_2[..., 2] = pred_trans_2[..., 2] * tx_max
    pred_trans_2[..., 3] = pred_trans_2[..., 3] * ty_max
    
    
    # create gt values FOR MATCHING PAIRS

    angle_diff = t_match_obj.angle - source_obj.angle
    if angle_diff < 0:
        angle_diff = 360 + angle_diff
    radian = math.radians(angle_diff)
    # create GT tx and ty
    gt_tx = t_match_obj.center_x - source_obj.center_x
    gt_ty = t_match_obj.center_y - source_obj.center_y
    gt_trans = torch.tensor([math.cos(radian), math.sin(radian), gt_tx, gt_ty])
    org_img = torch.unsqueeze(cropper(map_image, source_obj),0)
    trans_img = torch.unsqueeze(cropper(map_image, t_match_obj),0)
    
    print(f"Current source angle: {source_obj.angle}")
    print(f"Current target angle: {t_match_obj.angle}")
    print(f"Current angle diff: {angle_diff}")
    print(f"Current source cx: {source_obj.center_x}")
    print(f"Current source cy: {source_obj.center_y}")
    print(f"Current target cx: {t_match_obj.center_x}")
    print(f"Current target cy: {t_match_obj.center_y}")
    print(f"Current center diff (tx, ty): {gt_tx},{gt_ty}")
    
    vis_prediction(map_image=map_image, source_rect=source_obj,org_img=org_img, trans_img=trans_img,
                   gt_match=1,gt_trans=gt_trans, 
                   prediction_affine=pred_trans,prediction_match=pred_match, gt_angle_diff=angle_diff, idx=i)
