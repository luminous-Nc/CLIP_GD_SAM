import numpy as np
import cv2
import os


def calculate_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, np.logical_not(mask2)) # Black and white are reversed in ground and result
    union = np.logical_or(mask1, np.logical_not(mask2))
    iou = np.sum(intersection) / np.sum(union)
    return iou


if __name__ == "__main__":
    result_folder = ".\\coca\\coca_result"
    ground_folder = ".\\coca\\coca_ground"

    categories = os.listdir(ground_folder)

    ious_per_category = []

    for category in categories:
        category_result_path = os.path.join(result_folder, category)
        category_ground_path = os.path.join(ground_folder, category)

        result_images = os.listdir(category_result_path)
        ground_images = os.listdir(category_ground_path)

        ious = []

        for image_name in ground_images:
            ground_mask_path = os.path.join(category_ground_path, image_name)

            corresponding_result_name = image_name[:-4] + '.jpg'
            result_mask_path = os.path.join(category_result_path, corresponding_result_name)

            if not os.path.exists(result_mask_path):
                iou = 0
                ious.append(iou)
                continue
            else:
                ground_mask = cv2.imread(ground_mask_path, cv2.IMREAD_GRAYSCALE)
                result_mask = cv2.imread(result_mask_path, cv2.IMREAD_GRAYSCALE)
                iou = calculate_iou(ground_mask, result_mask)
                ious.append(iou)

        category_average_iou = np.mean(ious)
        ious_per_category.append(category_average_iou)
        print("Category: {:<15} Average IoU: {:.4f}".format(category, category_average_iou))

    mean_iou = np.mean(ious_per_category)
    print(f"Mean mIOU: {mean_iou}")