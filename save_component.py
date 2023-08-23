from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def save_data(word, boundingbox, mask, output_path, image_file_name):
    # print(f"word:{word}")
    # print(f"boundingbox:{boundingbox}")
    # print(f"output_path:{output_path}")
    # print(f"image_filename:{image_file_name}")
    mask_array = np.logical_not(np.array(mask)).astype(np.uint8) * 255
    result_category_path = os.path.join(output_path, word)
    os.makedirs(result_category_path, exist_ok=True)
    result_image_path = os.path.join(result_category_path, image_file_name)
    cv2.imwrite(result_image_path, mask_array)

    # print(boundingbox["bb_abs_cxywh"])
    result_txt_path = os.path.join(result_category_path,image_file_name) + ".txt"
    with open(result_txt_path,"w") as txt_file:
        txt_file.write(str(boundingbox["bb_abs_cxywh"]))

    print(f"save to {result_image_path}")



