import os
import datetime

import cv2
import numpy as np
import pandas as pd
import openpyxl

from clip_module import get_clip_model, get_word_from_clip
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

category_valid_id = [0, 25, 29, 58, 84, 13, 59, 82, 24, 8, 37, 15, 65, 36, 40, 70, 66, 49, 56, 30, 73]


def calculate_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2)  # Black and white are reversed in ground and result
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def find_corresponding_label_ids_file(image_info, ground_folder):
    image_name = os.path.splitext(os.path.basename(image_info['address']))[0]  # 提取图像名称
    label_ids_file = image_name + ".png"
    label_ids_file_path = os.path.join(ground_folder, label_ids_file)
    return label_ids_file_path


def collect_image_info(category_result_folder):
    image_info_list = []

    for file_name in os.listdir(category_result_folder):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            file_path = os.path.join(category_result_folder, file_name)
            image_info = {'address': file_path, 'image_id': file_name[:-4]}
            image_info_list.append(image_info)
    return image_info_list


def display_pixels_by_category(label_ids_path, category_id, category):
    label_ids = cv2.imread(label_ids_path, cv2.IMREAD_UNCHANGED)

    category_pixels = (label_ids == category_id)
    segmented_image = np.zeros_like(label_ids)
    segmented_image[category_pixels] = label_ids[category_pixels]

    resized_image = cv2.resize(segmented_image, (1024, 726))

    cv2.imshow(f"category:{category} image:{label_ids_path}", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_ground_image_path(image_info, result_folder):
    image_name = os.path.splitext(os.path.basename(image_info['address']))[0]  # 提取图像名称
    label_name_file = image_name + f"_{image_info['category']}" + ".png"
    label_ids_file_path = os.path.join(result_folder, label_name_file)
    return label_ids_file_path


def get_id_and_categories(id_array):
    result_string = ""
    for id in id_array:
        category_name = next((k for k, v in category_id_mapping.items() if v == id), None)
        result_string += f"{id}-{category_name}, "

    result_string = result_string.rstrip(", ")
    return result_string


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
    masks_folder = ".\\..\\dataset\\food_103\\pure_masks_simple - 0.35"
    compare_result_folder = ".\\..\\dataset\\food_103\\ppt_result"
    ground_info_list = []
    category_valid_id = [0, 25, 29, 58, 84, 13, 59, 82, 24, 8, 37, 15, 65, 36, 40, 70, 66, 49, 56, 30, 73]
    image_info_list = collect_image_info(masks_folder)

    for image_info in image_info_list:
        file_name = image_info["address"].split("\\")[-1]
        corresponding_label_ids_file = find_corresponding_label_ids_file(image_info, ground_folder)

        ground_truth_mask = Image.open(corresponding_label_ids_file)
        result_mask = Image.open(image_info["address"])
        width, height = result_mask.size
        image_file_name = os.path.basename(image_info["address"])
        ppt_image_path = os.path.join(compare_result_folder, image_file_name)

        result_image = Image.new("RGB", (width, height))

        for y in range(height):
            for x in range(width):
                mask_pixel = result_mask.getpixel((x, y))[0]
                gt_pixel = ground_truth_mask.getpixel((x, y))
                if mask_pixel == 0 and gt_pixel == 0:  # If both are background
                    result_image.putpixel((x, y), (0, 0, 0))  # Black
                elif gt_pixel in category_valid_id:  # Ground truth is valid  Mask must valid
                    if gt_pixel == mask_pixel:  # Both are same category correct
                        result_image.putpixel((x, y),
                                              (mask_pixel, mask_pixel, mask_pixel))
                    elif mask_pixel == 0 and gt_pixel != 0:  # mask not segment but ground truth has
                        result_image.putpixel((x, y), (255, 0, 0))  # Red
                    elif mask_pixel != 0 and gt_pixel != 0:  # Incorrect category
                        result_image.putpixel((x, y), (255, 255, 0))  # Yellow
                    elif mask_pixel != 0 and gt_pixel == 0:  # mask segment but don't in ground
                        result_image.putpixel((x, y), (0, 0, 255))  # Blue
                elif gt_pixel not in category_valid_id:
                    if mask_pixel == 0 and gt_pixel != 0:  # mask not segment and ground truth seen as not segment
                        result_image.putpixel((x, y), (0, 0, 0))  # Black
                    elif mask_pixel != 0 and gt_pixel != 0:  # Incorrect category but ground truth is beyond arrange
                        result_image.putpixel((x, y), (255, 255, 0))  # Yellow
        result_image.save(ppt_image_path)
        print(f"save to {ppt_image_path}")
