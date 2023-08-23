from PIL import Image

import torch
import cv2
import numpy as np

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
import GroundingDINO.groundingdino.datasets.transforms as T

from huggingface_hub import hf_hub_download

# Use this command for evaluate the GLIP-T model
config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filename = "groundingdino_swinb_cogcoor.pth"


def load_model_hf(model_config_path, repo_id, filename, device='cuda'):
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cuda')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None)  # 3, h, w
    return init_image, image


def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None)  # 3, h, w
    return image


model = load_model_hf(config_file, ckpt_repo_id, ckpt_filename)


def cxcywh_to_xyxy(image_PIL, cxcywh_box):
    image_width, image_height = image_PIL.size

    cx, cy, w, h = cxcywh_box

    left = int((cx - w / 2) * image_width)
    top = int((cy - h / 2) * image_height)
    right = int((cx + w / 2) * image_width)
    bottom = int((cy + h / 2) * image_height)

    xyxy_box = [left, top, right, bottom]
    return xyxy_box


def relative_to_absolute(image_PIL, cxcywh_box):
    image_width, image_height = image_PIL.size
    cx, cy, w, h = cxcywh_box
    abs_cx = cx * image_width
    abs_cy = cy * image_height
    abs_w = w * image_width
    abs_h = h * image_height
    cxcywh_box = [abs_cx, abs_cy, abs_w, abs_h]
    return cxcywh_box


def get_bb_from_grounding_dino(image_PIL, describe):
    init_image = image_PIL.convert("RGB")
    grounding_caption = describe
    box_threshold = 0.3
    text_threshold = 0.25

    _, image_tensor = image_transform_grounding(init_image)
    image_pil: Image = image_transform_grounding_for_vis(init_image)

    boxes, logits, phrases = predict(model, image_tensor, grounding_caption, box_threshold, text_threshold,
                                     device='cuda')
    if len(boxes) > 0:
        # annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)

        # cv2.imwrite('a.jpg', annotated_frame)
        # print(boxes, logits, phrases)
        bb_xyxy = cxcywh_to_xyxy(image_PIL, boxes.numpy()[0])
        bb_abs_cxywh = relative_to_absolute(image_PIL, boxes.numpy()[0])
        return {"bb_xyxy": bb_xyxy, "bb_abs_cxywh": bb_abs_cxywh}
    else:
        return False

