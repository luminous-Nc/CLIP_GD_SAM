import os

from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2
import torch

from global_setting import sam_model_type, sam_checkpoint, sam_device


class SAMModel:
    def __init__(self):
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=sam_device)
        self.predictor = SamPredictor(sam)

    def get_mask_list(self, image_PIL, bounding_box_list):
        image = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)
        self.predictor.set_image(image)
        mask_list = []
        for bounding_box in bounding_box_list:
            input_box = np.array(bounding_box['bb_xyxy'])
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            mask_list.append(masks[0])
            # ToDo Choose highest score masks
        return mask_list
