from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

colors = [
    [255, 0, 0],  # Blue
    [128, 0, 128],  # Purple
    [0, 0, 0],  # Black
    [0, 0, 255],    # Red
    [0, 165, 255],  # Orange
    [0, 255, 255],  # Yellow
    [0, 128, 0],    # Green
    [255, 255, 0],  #
    [255, 255, 255] # White
]

def save_data(word, boundingbox_list, mask_list, output_path, image_file_name, image_path ,annotated_image):
    # print(f"word:{word}")
    # print(f"boundingbox:{boundingbox}")
    # print(f"output_path:{output_path}")
    # print(f"image_filename:{image_file_name}")

    # original_image = cv2.imread(image_path)
    original_image = annotated_image


    result_image = original_image.copy()
    for i, mask in enumerate(mask_list):
        color = colors[i % len(colors)]
        color_mask = np.zeros_like(original_image)
        color_mask[:, :] = color  # B,G,R
        mask_rgb = np.where(mask[:, :, np.newaxis], color_mask, 0)
        result_image = cv2.add(result_image, mask_rgb)

    result_category_path = os.path.join(output_path, word)
    os.makedirs(result_category_path, exist_ok=True)
    result_image_path = os.path.join(result_category_path, image_file_name)
    cv2.imwrite(result_image_path, result_image)

    # # print(boundingbox["bb_abs_cxywh"])
    # result_txt_path = os.path.join(result_category_path,image_file_name) + ".txt"
    # with open(result_txt_path,"w") as txt_file:
    #     txt_file.write(str(boundingbox["bb_abs_cxywh"]))

    print(f"save to {result_image_path}")
