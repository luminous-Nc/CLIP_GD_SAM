import os

from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

colors = [
    [255, 0, 0],  # Blue
    [128, 0, 128],  # Purple
    [0, 0, 0],  # Black
    [0, 0, 255],  # Red
    [0, 165, 255],  # Orange
    [0, 255, 255],  # Yellow
    [0, 128, 0],  # Green
    [255, 255, 0],  #
    [255, 255, 255]  # White
]


def get_mask_predictor():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def get_input_boxes_from_bounding_boxes(predictor, bounding_box_list):
    input_boxes = []
    for bounding_box in bounding_box_list:
        bb_abs_xyxy = bounding_box["bb_xyxy"]
        input_boxes.append(bb_abs_xyxy)
    input_boxes = torch.tensor(input_boxes, device=predictor.device)
    return input_boxes


def get_mask_from_sam(predictor, image_PIL, bounding_box_list,image_path):
    image = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)
    predictor.set_image(image)
    mask_list = []
    for bounding_box in bounding_box_list:
        input_box = np.array(bounding_box['bb_xyxy'])
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        mask_list.append(masks[0])

    return mask_list


def get_mask_from_sam_with_boxes(predictor, image_PIL, bounding_box_list, image_path):
    input_boxes = get_input_boxes_from_bounding_boxes(predictor, bounding_box_list)
    image = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)
    predictor.set_image(image)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )


    # for i, mask in enumerate(masks):
    #     mask = mask.cpu().numpy()
    #     mask = np.squeeze(mask)
    #     color = [128, 0, 128]
    #     color_mask = np.zeros_like(image)
    #     color_mask[:, :] = color  # B,G,R
    #     mask_rgb = np.where(mask[:, :, np.newaxis], color_mask, 0)
    #     image_file_name = os.path.basename(image_path)
    #     cv2.imwrite(f"{image_file_name}-mask-new-{i}.jpg", mask_rgb)

    return masks

