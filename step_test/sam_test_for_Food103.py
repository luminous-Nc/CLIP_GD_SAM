import os

import cv2
import numpy as np

from clip_component import get_clip_model, get_word_from_clip
from PIL import Image

word_list = ["background", "candy", "egg tart", "french fries", "chocolate", "biscuit",
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


def calculate_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, np.logical_not(mask2))  # Black and white are reversed in ground and result
    union = np.logical_or(mask1, np.logical_not(mask2))
    iou = np.sum(intersection) / np.sum(union)
    return iou


def collect_image_info(category_result_folder, category):
    image_info_list = []

    category_id = word_list.index(category)
    for file_name in os.listdir(category_result_folder):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            file_path = os.path.join(category_result_folder, file_name)
            image_info = {'address': file_path, 'category': category, 'category_id': category_id}
            image_info_list.append(image_info)
    return image_info_list


def find_corresponding_label_ids_file(image_info, ground_folder):
    image_name = os.path.splitext(os.path.basename(image_info['address']))[0]  # 提取图像名称
    label_ids_file = image_name + ".png"
    label_ids_file_path = os.path.join(ground_folder, label_ids_file)
    return label_ids_file_path


def display_pixels_by_category(label_ids_path, category_id, category):
    label_ids = cv2.imread(label_ids_path, cv2.IMREAD_UNCHANGED)

    category_pixels = (label_ids == category_id)
    segmented_image = np.zeros_like(label_ids)
    segmented_image[category_pixels] = label_ids[category_pixels]

    resized_image = cv2.resize(segmented_image, (1024, 726))

    cv2.imshow(f"category:{category} image:{label_ids_path}", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    ground_folder = ".\\..\\dataset\\food_103\\ground_truths"
    result_folder = ".\\..\\dataset\\food_103\\pure_masks"

    categories = os.listdir(result_folder)
    ious_per_category = []

    for category in categories:
        ious = []
        category_result_folder = os.path.join(result_folder, category)
        image_info_list = collect_image_info(category_result_folder, category)
        if len(image_info_list) == 0:
            print("Category: {:<15} Average IoU: {:<15}".format(category, '0 CLIP dont recognize this category'))
            ious_per_category.append(0)
            continue
        else:
            for image_info in image_info_list:
                corresponding_label_ids_file = find_corresponding_label_ids_file(image_info, ground_folder)
                category_id = image_info['category_id']
                label_ids = cv2.imread(corresponding_label_ids_file, cv2.IMREAD_UNCHANGED)

                # display_pixels_by_category(corresponding_label_ids_file, category_id, category)

                ground_pixels = (label_ids != category_id)
                ground_mask = np.zeros_like(label_ids)
                ground_mask[ground_pixels] = 255

                result_mask = cv2.imread(image_info['address'], cv2.IMREAD_GRAYSCALE)
                iou = calculate_iou(ground_mask, result_mask)
                ious.append(iou)
            category_average_iou = np.mean(ious)
            ious_per_category.append(category_average_iou)
            print("Category: {:<15} Average IoU: {:.4f}%".format(category, category_average_iou * 100))
    mean_iou = np.mean(ious_per_category)
    print(f"Mean mIOU: {mean_iou}")
