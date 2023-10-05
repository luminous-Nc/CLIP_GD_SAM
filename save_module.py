import numpy as np
import cv2
import os

from util import get_word_id

colors = [  # BGR format
    [255, 0, 0],  # Blue
    [128, 0, 128],  # Purple
    [0, 0, 0],  # Black
    [0, 0, 255],  # Red
    [0, 165, 255],  # Orange
    [0, 255, 255],  # Yellow
    [0, 128, 0],  # Green
    [255, 255, 0],  # Shallow blue
    [255, 255, 255]  # White
]


def save_data_single_category(image_PIL, word, boundingbox_list, mask_list, image_path, annotated_image,
                              task_info, task_args):
    if task_args.bb:
        original_image = annotated_image
    else:
        original_image = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)

    # ToDo: Make result more easy to see and compare
    if task_args.result:
        result_image = original_image.copy()

        for i, mask in enumerate(mask_list):
            color = colors[i % len(colors)]
            color_mask = np.zeros_like(original_image)
            color_mask[:, :] = color  # B,G,R
            mask_rgb = np.where(mask[:, :, np.newaxis], color_mask, 0)
            result_image = cv2.add(result_image, mask_rgb)

        output_path = task_info["output_path"]
        result_category_path = os.path.join(output_path, word)
        image_file_name = os.path.basename(image_path)
        result_image_path = os.path.join(result_category_path, image_file_name)
        cv2.imwrite(result_image_path, result_image)
        print(f"save result to {result_image_path}")

    output_pure_path = task_info["output_pure_path"]
    if task_args.bb_txt:
        # Cut out extension name, get "image name" + txt
        image_file_name = os.path.basename(image_path)
        txt_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        result_category_path = os.path.join(output_pure_path, word)
        result_txt_path = os.path.join(result_category_path, txt_file_name)

        with open(result_txt_path, "w") as txt_file:
            for boundingbox in boundingbox_list:
                bb_abs_cxywh = boundingbox["bb_abs_cxywh"]
                if "selected_word_list" in task_info:
                    word_id = task_info["word_id_mapping"]["word"]
                else:
                    word_id = get_word_id(word, task_info["word_list"])
                input_line = f"{word_id} {bb_abs_cxywh[0]} {bb_abs_cxywh[1]} {bb_abs_cxywh[2]} {bb_abs_cxywh[3]} \n"
                txt_file.write(input_line)
        print(f"save bounding boxes to {result_txt_path}")

    if task_args.pure_mask:
        pure_mask_image = np.zeros_like(original_image)
        for i, mask in enumerate(mask_list):
            color = [word_id, word_id, word_id]
            color_mask = np.zeros_like(original_image)
            color_mask[:, :] = color  # B,G,R  CV2 use BGR as default
            mask_rgb = np.where(mask[:, :, np.newaxis], color_mask, 0)
            pure_mask_image = cv2.add(pure_mask_image, mask_rgb)

        pure_category_path = os.path.join(output_pure_path, word)
        os.makedirs(pure_category_path, exist_ok=True)
        image_file_name = os.path.basename(image_path)
        image_mask_name = os.path.splitext(image_file_name)[0] + ".png"
        pure_mask_path = os.path.join(pure_category_path, image_mask_name)
        cv2.imwrite(pure_mask_path, pure_mask_image)
        print(f"save pure mask to {pure_mask_path}")


def save_data_multiple_category(image_PIL, boundingbox_list, mask_list, image_path, annotated_image,
                                task_info, task_args):
    if task_args.bb:
        original_image = annotated_image
    else:
        original_image = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)
    # ToDo: Make result more easy to see and compare
    if task_args.result:
        result_image = original_image.copy()

        for i, mask in enumerate(mask_list):
            color = colors[i % len(colors)]
            color_mask = np.zeros_like(original_image)
            color_mask[:, :] = color  # B,G,R
            mask_rgb = np.where(mask[:, :, np.newaxis], color_mask, 0)
            result_image = cv2.add(result_image, mask_rgb)

        output_path = task_info["output_path"]
        image_file_name = os.path.basename(image_path)
        result_image_path = os.path.join(output_path, image_file_name)
        cv2.imwrite(result_image_path, result_image)
        print(f"save result to {result_image_path}")
    output_pure_path = task_info["output_pure_path"]
    if task_args.bb_txt:
        output_pure_path = task_info["output_pure_path"]
        # Cut out extension name, get "image name" + txt
        image_file_name = os.path.basename(image_path)
        txt_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        result_txt_path = os.path.join(output_pure_path, txt_file_name)

        with open(result_txt_path, "w") as txt_file:
            for boundingbox in boundingbox_list:
                bb_abs_cxywh = boundingbox["bb_abs_cxywh"]
                if "selected_word_list" in task_info:
                    word_id = task_info["word_id_mapping"][boundingbox["word"]]
                else:
                    word_id = get_word_id(boundingbox["word"], task_info["word_list"])
                input_line = f"{word_id} {bb_abs_cxywh[0]} {bb_abs_cxywh[1]} {bb_abs_cxywh[2]} {bb_abs_cxywh[3]} \n"
                txt_file.write(input_line)
        print(f"save bounding boxes to {result_txt_path}")

    if task_args.pure_mask:
        pure_mask_image = np.zeros_like(original_image)

        for i, mask in enumerate(mask_list):
            boundingbox = boundingbox_list[i]
            if "selected_word_list" in task_info:
                word_id = task_info["word_id_mapping"][boundingbox["word"]]
            else:
                word_id = get_word_id(boundingbox["word"], task_info["word_list"])
            pure_mask_image[mask] = [word_id, word_id, word_id]

        image_file_name = os.path.basename(image_path)
        image_mask_name = os.path.splitext(image_file_name)[0] + ".png"
        pure_mask_path = os.path.join(output_pure_path, image_mask_name)
        cv2.imwrite(pure_mask_path, pure_mask_image)
        print(f"save pure mask to {pure_mask_path}")
