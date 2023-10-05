import os
import datetime

import cv2
import numpy as np
import pandas as pd
import openpyxl

from clip_module import get_clip_model, get_word_from_clip, get_word_list_from_clip
from PIL import Image

full_103 = ["background", "candy", "egg tart", "french fries", "chocolate", "biscuit",
            "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine", "milkshake",
            "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew", "dried cranberries",
            "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado",
            "banana", "strawberry",
            "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig",
            "pineapple",
            "grape", "kiwi", "melon", "orange", "watermelon",
            "steak", "pork", "chicken duck", "sausage", "fried meat",
            "lamb", "sauce", "crab", "fish", "shellfish",
            "shrimp", "soup", "bread", "corn", "hamburg",
            "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles",
            "rice", "pie", "tofu", "eggplant", "potato",
            "garlic", "cauliflower", "tomato", "kelp", "seaweed",
            "spring onion", "rape", "ginger", "okra", "lettuce",
            "pumpkin", "cucumber", "white radish", "carrot", "asparagus",
            "bamboo shoots", "broccoli", "celery stick", "cilantro mint", "snow peas",
            "cabbage", "bean sprouts", "onion", "pepper", "green beans",
            "French beans", "king oyster mushroom", "shiitake", "enoki mushroom",
            "oyster mushroom", "white button mushroom", "salad", "other ingredients"
            ]
category_id_mapping = {
    'apple': 25,
    'banana': 29,
    'bread': 58,
    'carrot': 84,
    'coffee': 13,
    'corn': 59,
    'cucumber': 82,
    'egg': 24,
    'ice cream': 8,
    'lemon': 37,
    'milk': 15,
    'noodles': 65,
    'peach': 36,
    'pineapple': 40,
    'potato': 70,
    'rice': 66,
    'sausage': 49,
    'shrimp': 56,
    'strawberry': 30,
    'tomato': 73,
}

valid_id = [0, 25, 29, 58, 84, 13, 59, 82, 24, 8, 37, 15, 65, 36, 40, 70, 66, 49, 56, 30, 73]

word_list = ["apple", "banana", "bread", "carrot", "coffee", "corn", "cucumber", "egg", "ice cream",
             "lemon", "milk", "noodles", "peach", "pineapple", "potato", "rice", "sausage", "shrimp",
             "strawberry", "tomato"
             ]


def find_corresponding_label_ids_file(image_info, file_folder):
    image_name = os.path.splitext(os.path.basename(image_info['address']))[0]  # 提取图像名称
    label_ids_file = image_name + ".png"
    label_ids_file_path = os.path.join(file_folder, label_ids_file)
    return label_ids_file_path


def collect_image_info(category_result_folder):
    image_info_list = []

    for file_name in os.listdir(category_result_folder):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            file_path = os.path.join(category_result_folder, file_name)
            image_info = {'address': file_path, 'image_id': file_name[:-4]}
            image_info_list.append(image_info)
    return image_info_list


def get_id_and_categories_ground(id_array):
    result_string = ""
    for id in id_array:
        category_name = full_103[id]
        result_string += f"{id}-{category_name}, "

    result_string = result_string.rstrip(", ")
    return result_string


if __name__ == "__main__":
    # threshold = 0.35
    threshold = 0.50
    # threshold = 0.65
    ground_folder = ".\\..\\dataset\\food_103\\ground_truths"
    # masks_folder = f".\\..\\dataset\\food_103\\pure_masks_simple - {threshold}"
    image_folder = ".\\..\\dataset\\food_103\\images"

    current_time = datetime.datetime.now()
    time_format = current_time.strftime("%H%M%S")
    directory = "compare_result_clip"
    os.makedirs(directory, exist_ok=True)  # 创建目录（如果不存在）
    result_txt_name = f"{directory}/result_test_{time_format}.txt"
    clip_model, clip_preprocess, device, text_features = get_clip_model(word_list)
    print(result_txt_name)
    with open(result_txt_name, "w") as txt_file:
        txt_file.write(f"Begin to compare threshold: {threshold} \n")
        image_list = collect_image_info(image_folder)
        for image_info in image_list:
            file_name = image_info["address"].split("\\")[-1]
            corresponding_label_ids_file = find_corresponding_label_ids_file(image_info, ground_folder)
            ground_truth_array = np.array(Image.open(corresponding_label_ids_file))
            unique_ids_ground = [x for x in np.unique(ground_truth_array) if x in valid_id and x != 0]

            input_line = f"{file_name} \n"
            txt_file.write(input_line)
            txt_file.write(f"  ground_truth_categories: {get_id_and_categories_ground(unique_ids_ground)} \n")

            image_PIL = Image.open(image_info["address"])
            result_list = get_word_list_from_clip(image_PIL, clip_model, clip_preprocess, device, word_list,
                                                  text_features)
            for one_word in result_list:
                text_input = one_word["text_input"]
                similarity = one_word["similarity"]
                txt_file.write(f"    word:  {text_input}  similarity  {similarity} \n")
            # print(word_list)
