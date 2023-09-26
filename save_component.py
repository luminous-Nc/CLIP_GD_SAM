from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

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


def save_data(id, word, boundingbox_list, mask_list, output_path, output_pure_path, image_path, annotated_image,
              task_args):
    # print(f"id:{id}"
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
    image_file_name = os.path.basename(image_path)
    result_image_path = os.path.join(result_category_path, image_file_name)
    cv2.imwrite(result_image_path, result_image)
    print(f"save result to {result_image_path}")

    if task_args.bounding_box:
        file_name_without_extension = os.path.splitext(image_file_name)[0]  # 去除扩展名，得到 "image"
        result_txt_path = os.path.join(result_category_path, file_name_without_extension) + ".txt"

        with open(result_txt_path, "w") as txt_file:
            for boundingbox in boundingbox_list:
                bb_abs_cxywh = boundingbox["bb_abs_cxywh"]
                input_line = f"{id} {bb_abs_cxywh[0]} {bb_abs_cxywh[1]} {bb_abs_cxywh[2]} {bb_abs_cxywh[3]} \n"
                txt_file.write(input_line)
        print(f"save bounding boxes to {result_txt_path}")

    if task_args.pure_mask:
        pure_mask_image = np.zeros_like(original_image)
        for i, mask in enumerate(mask_list):
            print(f"id:{id} of instance {i}")
            color = [id, id, id]
            color_mask = np.zeros_like(original_image)
            color_mask[:, :] = color  # B,G,R
            mask_rgb = np.where(mask[:, :, np.newaxis], color_mask, 0)
            pure_mask_image = cv2.add(pure_mask_image, mask_rgb)

        pure_category_path = os.path.join(output_pure_path, word)
        os.makedirs(pure_category_path, exist_ok=True)
        image_file_name = os.path.basename(image_path)
        pure_mask_path = os.path.join(pure_category_path, image_file_name)
        cv2.imwrite(pure_mask_path, pure_mask_image)
        print(f"save pure mask to {pure_mask_path}")


