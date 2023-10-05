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
    # masks_folder = f".\\..\\dataset\\food_103\\pure_masks_simple - {threshold}"
    masks_folder = f".\\..\\dataset\\food_103\\pure_masks_simple"
    category_scores = {}
    accuracy_scores = {}

    total_icus = []
    total_accuracy = []
    result_ids = []

    current_time = datetime.datetime.now()
    time_format = current_time.strftime("%H%M%S")
    result_txt_name = f"compare_result/result_test_{time_format}.txt"
    result_excel_name = f"compare_result/result_test_{time_format}.xlsx"

    result_category_excel_name = f"compare_result/result_category_{time_format}.xlsx"

    with open(result_txt_name, "w") as txt_file:
        image_info_list = collect_image_info(masks_folder)
        txt_file.write(f"With threshold {threshold} There are {len(image_info_list)} images in total\n")
        for image_info in image_info_list:
            file_name = image_info["address"].split("\\")[-1]
            input_line = f"\n{file_name} \n"
            txt_file.write(input_line)

            image_accuracy = []
            image_ious = []

            corresponding_label_ids_file = find_corresponding_label_ids_file(image_info, ground_folder)

            ground_truth_array = np.array(Image.open(corresponding_label_ids_file))
            result_array = np.array(Image.open(image_info["address"]))

            unique_ids_ground = np.unique(ground_truth_array)
            unique_ids_result = np.unique(result_array)
            input_line = f"  ground_truth_categories: {get_id_and_categories_ground(unique_ids_ground)} \n"
            txt_file.write(input_line)
            input_line = f"  result_categories: {get_id_and_categories_ground(unique_ids_result)} \n"
            txt_file.write(input_line)
            for one_category_id in unique_ids_result:
                if one_category_id == 0:
                    continue
                one_category_name = full_103[one_category_id]

                if one_category_name not in category_scores:
                    category_scores[one_category_name] = []
                    accuracy_scores[one_category_name] = []

                if one_category_id in unique_ids_ground:
                    image_accuracy.append(100)
                    accuracy_scores[one_category_name].append(100)
                else:
                    image_accuracy.append(0)
                    accuracy_scores[one_category_name].append(0)

                ground_pixels = (ground_truth_array == one_category_id)
                ground_mask = np.zeros_like(ground_truth_array)
                ground_mask[ground_pixels] = one_category_id

                result_pixels = (result_array == one_category_id)
                result_mask = np.zeros_like(result_array)
                result_mask[result_pixels] = one_category_id
                result_mask = np.mean(result_mask, axis=2)

                iou_for_one_category = calculate_iou(ground_mask, result_mask)
                image_ious.append(iou_for_one_category)
                category_scores[one_category_name].append(iou_for_one_category)
                input_line = f"    IoU for {one_category_id} - {full_103[one_category_id]} : {iou_for_one_category} \n"
                txt_file.write(input_line)
            if len(image_ious) == 0:
                continue
            image_average_iou = np.mean(image_ious)
            image_average_accuracy = np.mean(image_accuracy)

            total_icus.append(image_average_iou)
            total_accuracy.append(image_average_accuracy)
            result_ids.append(file_name)

            input_line = f"average iou:{image_average_iou} average_accuracy:{image_average_accuracy}\n"
            txt_file.write(input_line)

        df = pd.DataFrame({'image ID': result_ids, 'IoU': total_icus, 'Detect Accuracy': total_accuracy})
        df.to_excel(result_excel_name)
        mean_iou = np.mean(total_icus)
        mean_accuracy = np.mean(total_accuracy)
        print(f"Mean mIOU: {mean_iou}")
        print(f"Mean accuracy: {mean_accuracy}")
        input_line = f"average mIOU: {mean_iou} \n\n"
        txt_file.write(input_line)
        input_line = f"average Accuracy: {mean_accuracy} \n\n"
        txt_file.write(input_line)

        txt_file.write("category IOU\n")

        excel_category = []
        excel_total_icus = []
        excel_total_accuracy = []
        for category, scores in category_scores.items():
            excel_category.append(category)
            average = sum(scores) / len(scores)
            excel_total_icus.append(average)
            input_line = f"{category}   {average} \n"
            txt_file.write(input_line)

        for category, scores in accuracy_scores.items():
            average = sum(scores) / len(scores)
            excel_total_accuracy.append(average)
        df = pd.DataFrame(
            {'image ID': excel_category, 'IoU': excel_total_icus, 'Detect Accuracy': excel_total_accuracy})
        df.to_excel(result_category_excel_name)
