import time
import argparse
from pipeline_config import task_mappings
from PIL import Image
from tqdm import tqdm

from clip_component import get_clip_model, get_word_from_clip
from grounding_component import get_bb_from_grounding_dino
from sam_component import get_mask_predictor, get_mask_from_sam
from save_component import save_data

import os


def process_one_image(image_path, clip_model, clip_preprocess, device, word_list, text_features, mask_predictor,
                      output_path, image_file_name):
    # print(f"process {image_path}")
    image_PIL = Image.open(image_path)

    word = get_word_from_clip(image_PIL, clip_model, clip_preprocess, device, word_list, text_features)
    # print(f"word:{word}")

    boundingbox_list, annotated_bb_image = get_bb_from_grounding_dino(image_PIL, word)
    if len(boundingbox_list) == 0:
        os.remove(image_path)
        print('Empty Detection')
        return
    print(f"bounding box number:{len(boundingbox_list)}")

    mask_list = get_mask_from_sam(mask_predictor, image_PIL, boundingbox_list)
    # print("mask done.")

    save_data(word, boundingbox_list, mask_list, output_path, image_file_name, image_path, annotated_bb_image)

    os.remove(image_path)


def USSPipeline(args):
    task_info = task_mappings.get(args.task)
    if task_info:
        word_list = task_info["word_list"]
        data_set_path = task_info["data_set_path"]
        output_path = task_info["output_path"]
        if args.binary_mask is not None:
            task_info["binary_mask"] = args.binary_mask
        binary_mask = task_info["binary_mask"]

        print(f"USS Pipeline Task: {args.task}")

        clip_model, clip_preprocess, device, text_features = get_clip_model(word_list)
        mask_predictor = get_mask_predictor()

        all_image_paths = []
        for root, dirs, files in os.walk(data_set_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', 'webp', '.gif')):
                    all_image_paths.append(os.path.join(root, file))

        with tqdm(total=len(all_image_paths), dynamic_ncols=True, desc=f"Generating {args.task} masks") as pbar:
            for image_path in all_image_paths:
                image_file_name = os.path.basename(image_path)
                start_time = time.time()
                process_one_image(image_path, clip_model, clip_preprocess, device, word_list, text_features,
                                  mask_predictor,
                                  output_path, image_file_name)
                end_time = time.time()
                elapsed_time = end_time - start_time
                pbar.set_postfix({"Time": f"{elapsed_time:.2f} s"})
                pbar.update(1)
        print(f"Task {args.task} Finished")
    else:
        print(f"Unknown taskname: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USS Pipeline parameters")
    parser.add_argument("--task", required=True, help="Specify the task name")
    parser.add_argument("--remove", type=bool, default=False, help="Specify whether to remove images after processing")
    parser.add_argument("--binary_mask", type=bool, help="Specify if generate binary mask (overrides task config)")

    args = parser.parse_args()
    USSPipeline(args)
