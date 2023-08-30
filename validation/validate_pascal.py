import numpy as np
import cv2
import os
from PIL import Image


def calculate_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, np.logical_not(mask2))  # Black and white are reversed in ground and result
    union = np.logical_or(mask1, np.logical_not(mask2))
    iou = np.sum(intersection) / np.sum(union)
    return iou


def find_files_with_suffix(folder_path, suffix):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                file_paths.append(os.path.join(root, file))
    return file_paths


def collect_image_info(category_result_folder, category):
    image_info_list = []
    category_id_mapping = {
        "aeroplane": 1,
         "bicycle": 2,
         "bird": 3,
         "boat": 4,
         "bottle": 5,
         "bus": 6,
         "car": 7,
         "cat": 8,
         "chair": 9,
         "cow": 10,
         "dining table": 11,
         "dog": 12,
         "horse": 13,
         "motorbike": 14,
         "person": 15,
         "potted plant": 16,
         "sheep": 17,
         "sofa": 18,
         "train": 19,
         "TV or monitor": 20
    }

    category_id = category_id_mapping.get(category, 0)
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


def display_pixels_by_category(label_ids_path, category_id):
    label_ids = cv2.imread(label_ids_path, cv2.IMREAD_UNCHANGED)

    category_pixels = (label_ids == category_id)
    segmented_image = np.zeros_like(label_ids)
    segmented_image[category_pixels] = label_ids[category_pixels]

    resized_image = cv2.resize(segmented_image, (1024, 726))

    cv2.imshow('Segmented Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_iou(prediction_mask, ground_truth_mask):
    intersection = np.logical_and(prediction_mask, ground_truth_mask)
    union = np.logical_or(prediction_mask, ground_truth_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou


if __name__ == "__main__":
    result_folder = ".\\pascal\\pascal_result"
    ground_folder = ".\\pascal\\pascal_ground"

    categories = os.listdir(result_folder)

    ious_per_category = []

    for category in categories:
        ious = []
        category_result_folder = os.path.join(result_folder, category)
        image_info_list = collect_image_info(category_result_folder, category)
        if len(image_info_list) == 0:
            print("Category: {:<15} Average IoU: {:<15}".format(category, '0 CLIP dont recognize this category'))
            # ious_per_category.append(0)
            continue
        else:
            for image_info in image_info_list:
                corresponding_label_ids_file = find_corresponding_label_ids_file(image_info, ground_folder)
                category_id = image_info['category_id']

                label_ids = Image.open(corresponding_label_ids_file).convert('P')

                label_ids = np.array(label_ids)
                ground_pixels = (label_ids != category_id)
                ground_mask = np.zeros_like(label_ids)
                ground_mask[ground_pixels] = 255

                cv2.imshow('Ground',ground_mask)


                result_mask = cv2.imread(image_info['address'], cv2.IMREAD_GRAYSCALE)
                cv2.imshow('Result',result_mask)
                cv2.waitKey(0)
                
                iou = calculate_iou(ground_mask, result_mask)
                ious.append(iou)
            category_average_iou = np.mean(ious)
            ious_per_category.append(category_average_iou)
            print("Category: {:<15} Average IoU: {:.4f}".format(category, category_average_iou))
    mean_iou = np.mean(ious_per_category)
    print(f"Mean mIOU: {mean_iou}")
