import time
import argparse
from pipeline_config import task_mappings
from PIL import Image, ImageFile
from tqdm import tqdm

from prepare_component import prepare_categories_and_create_dir
from clip_module import CLIPModel
from grounding_module import GDModel
from sam_module import SAMModel
from save_module import save_data_multiple_category

import os
from util import get_all_images_from_directory, file_exists_in_directory
from global_setting import clip_select_category_number


def process_one_image_multiple_category(image_path, clip_model, gd_model, sam_model, task_info, task_args):
    print(f"process {image_path}")
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Enable for loading truncated images
    image_PIL = Image.open(image_path)

    word_list_to_GD = clip_model.get_word_list_from_clip(image_PIL, clip_select_category_number)

    boundingbox_list, annotated_bb_image = gd_model.get_bb_list_from_image(image_PIL, word_list_to_GD)

    if len(boundingbox_list) == 0:
        print(f'Empty Detection for {word_list_to_GD}')
        return
    print(f"bounding box number:{len(boundingbox_list)}")

    mask_list = sam_model.get_mask_list(image_PIL, boundingbox_list)

    boundingbox_list, mask_list = clip_model.recheck_from_sam(boundingbox_list, mask_list, image_PIL)

    save_data_multiple_category(image_PIL, boundingbox_list, mask_list, image_path, annotated_bb_image, task_info,
                                task_args)


def USSPipeline(task_args):
    task_info = task_mappings.get(task_args.task)
    if task_info:
        word_list = task_info["word_list"]
        data_set_path = task_info["data_set_path"]
        output_path = task_info["output_path"]
        print(f"USS Pipeline Task: {task_args.task}")

        word_array = " . ".join(word_list)
        print(word_array)

        clip_model = CLIPModel(word_list, task_args.fine_tune)
        gd_model = GDModel()
        sam_model = SAMModel()

        all_image_paths = get_all_images_from_directory(data_set_path)

        with tqdm(total=len(all_image_paths), dynamic_ncols=True, desc=f"Generating {task_args.task} masks") as pbar:
            for image_path in all_image_paths:
                base_filename_to_check, _ = os.path.splitext(os.path.basename(image_path))

                if file_exists_in_directory(output_path, base_filename_to_check):
                    pbar.update(1)
                    continue

                start_time = time.time()
                process_one_image_multiple_category(image_path, clip_model, gd_model, sam_model, task_info, task_args)
                end_time = time.time()
                elapsed_time = end_time - start_time
                pbar.set_postfix({"Time": f"{elapsed_time:.2f} s"})
                pbar.update(1)
        print(f"Task {task_args.task} Finished")
    else:
        print(f"Unknown task name: {task_args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USS Pipeline parameters")
    parser.add_argument("--task", required=True, help="Specify the task name")
    parser.add_argument("--result", type=bool, default=True, help="Specify if generate result image")
    parser.add_argument("--pure_mask", type=bool, default=True,
                        help="Specify if generate pure mask image (overrides task config)")
    parser.add_argument("--bb", type=bool, default=True, help="Specify if draw bounding box on result image")
    parser.add_argument("--bb_txt", type=bool, default=False,
                        help="Specify whether need save bounding box information")
    parser.add_argument("--fine_tune", type=bool, default=False, help="Specify if use fine-tuned CLIP model")
    print("Program Start...")
    task_args = parser.parse_args()
    USSPipeline(task_args)
