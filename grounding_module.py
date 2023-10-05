import torch
import numpy as np

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
import GroundingDINO.groundingdino.datasets.transforms as T

from huggingface_hub import hf_hub_download
from global_setting import gd_config_file, gd_ckpt_repo_id, gd_ckpt_filename, box_threshold, text_threshold


class GDModel:
    def __init__(self, device='cuda'):
        args = SLConfig.fromfile(gd_config_file)
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=gd_ckpt_repo_id, filename=gd_ckpt_filename)
        checkpoint = torch.load(cache_file, map_location='cuda')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        self.model = model

    def get_bb_list_from_image(self, image_PIL, word_list):
        init_image = image_PIL.convert("RGB")  # must transfer to RGB from PIL

        grounding_caption = " . ".join(word_list)

        _, image_tensor = image_transform_grounding(init_image)

        boxes, logits, phrases = predict(self.model, image_tensor, grounding_caption, box_threshold, text_threshold,
                                         device='cuda')
        boxes_result = []
        annotated_frame = init_image
        if len(boxes) > 0:
            for index in range(boxes.shape[0]):
                bb_xyxy = cxcywh_to_xyxy(image_PIL, boxes.numpy()[index])
                bb_abs_cxywh = relative_to_absolute(image_PIL, boxes.numpy()[index])
                word = phrases[index]
                boxes_result.append({"word": word, "bb_xyxy": bb_xyxy, "bb_abs_cxywh": bb_abs_cxywh})
            annotated_frame = annotate(image_source=np.asarray(init_image), boxes=boxes, logits=logits, phrases=phrases)
            return boxes_result, annotated_frame

        else:
            return boxes_result, annotated_frame


def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None)  # 3, h, w
    return init_image, image


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
