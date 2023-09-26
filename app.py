import time
import argparse
from pipeline_config import task_mappings
from PIL import Image
from tqdm import tqdm

from prepare_component import prepare_categories_and_create_dir
from clip_component import get_clip_model, get_word_from_clip, get_word_id, resend_clip_to_check
from grounding_component import get_bb_from_grounding_dino
from sam_component import get_mask_predictor, get_mask_from_sam,get_mask_from_sam_with_boxes
from save_component import save_data

import os


def process_one_image(image_path, clip_model, clip_preprocess, device, word_list, text_features, mask_predictor,
                      output_path, output_pure_path, task_args):
    print(f"process {image_path}")
    image_PIL = Image.open(image_path)

    word = get_word_from_clip(image_PIL, clip_model, clip_preprocess, device, word_list, text_features)
    # print(f"word:{word}")
    word_id = get_word_id(word, word_list)

    boundingbox_list, annotated_bb_image = get_bb_from_grounding_dino(image_PIL, word)
    if len(boundingbox_list) == 0:
        os.remove(image_path)
        print('Empty Detection')
        return
    print(f"bounding box number:{len(boundingbox_list)}")

    mask_list = get_mask_from_sam(mask_predictor, image_PIL, boundingbox_list, image_path)
    # mask_list = get_mask_from_sam_with_boxes(mask_predictor, image_PIL, boundingbox_list,image_path)
    # print("mask done.")

    boundingbox_list, mask_list = resend_clip_to_check(word,boundingbox_list,mask_list,image_PIL,clip_model, clip_preprocess, device, word_list, text_features)

    save_data(word_id, word, boundingbox_list, mask_list, output_path, output_pure_path, image_path,
              annotated_bb_image, task_args)

    if task_args.remove:
        os.remove(image_path)


def USSPipeline(task_args):
    task_info = task_mappings.get(task_args.task)
    if task_info:
        word_list = task_info["word_list"]
        data_set_path = task_info["data_set_path"]
        output_path = task_info["output_path"]
        output_pure_path = task_info["output_pure_path"]
        print(f"USS Pipeline Task: {task_args.task}")

        clip_model, clip_preprocess, device, text_features = get_clip_model(word_list)
        mask_predictor = get_mask_predictor()

        all_image_paths = []
        for root, dirs, files in os.walk(data_set_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', 'webp')):
                    all_image_paths.append(os.path.join(root, file))

        with tqdm(total=len(all_image_paths), dynamic_ncols=True, desc=f"Generating {task_args.task} masks") as pbar:
            for image_path in all_image_paths:
                start_time = time.time()
                process_one_image(image_path, clip_model, clip_preprocess, device, word_list, text_features,
                                  mask_predictor, output_path, output_pure_path, task_args)
                end_time = time.time()
                elapsed_time = end_time - start_time
                pbar.set_postfix({"Time": f"{elapsed_time:.2f} s"})
                pbar.update(1)
        print(f"Task {task_args.task} Finished")
    else:
        print(f"Unknown taskname: {task_args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USS Pipeline parameters")
    parser.add_argument("--task", required=True, help="Specify the task name")
    parser.add_argument("--remove", type=bool, default=False, help="Specify whether to remove images after processing")
    parser.add_argument("--pure_mask", type=bool, default=False,
                        help="Specify if generate pure mask image (overrides task config)")
    parser.add_argument("--bounding_box", type=bool, default=False,
                        help="Specify whether need save bounding box information")
    print("Program Start...")
    task_args = parser.parse_args()
    prepare_categories_and_create_dir(task_args)
    USSPipeline(task_args)
