gd_config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
gd_ckpt_repo_id = "ShilongLiu/GroundingDINO"
gd_ckpt_filename = "groundingdino_swinb_cogcoor.pth"

sam_checkpoint = "sam_vit_h_4b8939.pth"
sam_model_type = "vit_h"
sam_device = "cuda"

box_threshold = 0.40  # choose the boxes whose highest similarities are higher than a box_threshold
text_threshold = 0.20  # extract the words whose similarities are higher than the text_threshold as predicted labels

clip_select_category_number = 5
