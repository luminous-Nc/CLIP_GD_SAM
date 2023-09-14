import time
import argparse
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

    boundingbox_list,annotated_bb_image = get_bb_from_grounding_dino(image_PIL, word)
    if len(boundingbox_list) == 0:
        os.remove(image_path)
        print('Empty Detection')
        return
    print(f"bounding box number:{len(boundingbox_list)}")

    mask_list = get_mask_from_sam(mask_predictor, image_PIL, boundingbox_list)
    # print("mask done.")

    save_data(word, boundingbox_list, mask_list, output_path, image_file_name, image_path,annotated_bb_image)

    os.remove(image_path)

def USSPipeline(taskname, remove_process):
    print('')

if __name__ == "__main__":


    word_list = [ "bacon", "baked beans", "beans and rice","beef lamb veal", "beverage", "biscuits",
  "breads", "breadsticks", "breakfast bars", "butter", "cakes", "candy", "cheese", "cheese sandwich","chili",
  "chips", "condiments and sauces", "cooked cereal", "cookies", "cornbread", "cottage cheese",
  "crackers", "cream cheese", "creamers", "creole", "dips", "dumpling", "egg rolls", "eggs", "fast food salads", "fish and seafood",
  "fish patty", "french fries", "french toast", "fruit", "fruit and vegetables", "gelatin", "gnocchi", "gravy", "ice cream", "lasagna",
  "luncheon meats", "mayonnaise", "mexican dishes", "olives", "pie", "pretzel", "rolls bagels buns",
  "sandwich", "shepherd pie", "sour cream", "spaghetti sauce", "sugar", "sweet breads","sweet breads pastries muffins", "syrups", "syrups lcings",
  "tortillas tacos shells", "trail and snack mix", "various salads", "vegetables", "waffle", "yogurt"]

    data_set_path = "./dataset/IBL_food/IBL Datasets"
    output_path = "./dataset/IBL_food/IBL results"

    # data_set_path = "./dataset/food_image/FoodImage"
    # output_path = "./dataset/food_image/food_image_result"

    clip_model, clip_preprocess, device, text_features = get_clip_model(word_list)
    mask_predictor = get_mask_predictor()

    all_image_paths = []
    for root, dirs, files in os.walk(data_set_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', 'webp', '.gif')):
                all_image_paths.append(os.path.join(root, file))

    with tqdm(total=len(all_image_paths), desc="Generating IBL Food masks") as pbar:
        for image_path in all_image_paths:
            image_file_name = os.path.basename(image_path)
            start_time = time.time()
            process_one_image(image_path, clip_model, clip_preprocess, device, word_list, text_features, mask_predictor,
                              output_path, image_file_name)
            end_time = time.time()
            elapsed_time = end_time - start_time
            pbar.set_postfix({"Time": f"{elapsed_time:.2f} s"})
            pbar.update(1)

    print("Task Finished")
