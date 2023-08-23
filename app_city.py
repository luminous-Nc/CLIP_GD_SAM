import json
import time

from PIL import Image
from tqdm import tqdm

from prepare_component import prepare_dataset_and_create_dir
from clip_component import get_clip_model, get_word_from_clip
from grounding_component import get_bb_from_grounding_dino
from sam_component import get_mask_predictor, get_mask_from_sam
from save_component import save_data

import os


def process_one_image(image_path, clip_model, clip_preprocess, device, word_list, mask_predictor, output_path,
                      image_file):
    print(f"process {image_path}")
    image_PIL = Image.open(image_path)

    word = get_word_from_clip(image_PIL, clip_model, clip_preprocess, device, word_list)
    # print(f"word:{word}")

    boundingbox = get_bb_from_grounding_dino(image_PIL, word)
    if boundingbox == False:
        os.remove(image_path)
        print('Empty Detection')
        return
    # print(f"bounding box:{boundingbox}")

    mask = get_mask_from_sam(mask_predictor, image_PIL, boundingbox)
    # print("mask done.")

    save_data(word, boundingbox, mask, output_path, image_file)

    os.remove(image_path)


if __name__ == "__main__":

    word_list = ['road', 'sidewalk', 'parking lot', 'rail track', 'person', 'rider', 'car', 'truck', 'bus', 'on rails',
                 'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence', 'guard rail', 'bridge',
                 'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'tree', 'terrain', 'sky']

    data_set_path = "dataset/cityscapes/city"
    output_path = "dataset/cityscapes/city_result"

    clip_model, clip_preprocess, device = get_clip_model()
    mask_predictor = get_mask_predictor()

    # Get all files list
    all_image_paths = []
    for root, dirs, files in os.walk(data_set_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                all_image_paths.append(os.path.join(root, file))
    # print(all_image_paths)

    with tqdm(total=len(all_image_paths), desc="Generating COCA masks") as pbar:
        for image_path in all_image_paths:
            image_file = os.path.basename(image_path)
            start_time = time.time()  # Begin time
            process_one_image(image_path, clip_model, clip_preprocess, device, word_list, mask_predictor, output_path,
                              image_file)
            end_time = time.time()  # End time
            elapsed_time = end_time - start_time
            pbar.set_postfix({"Time": f"{elapsed_time:.2f} s"})
            pbar.update(1)  # Update tqdm progress bar

    print("Task Finished")
