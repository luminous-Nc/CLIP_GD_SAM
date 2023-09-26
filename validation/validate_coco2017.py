import numpy as np
import cv2
import os
import json


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


def collect_image_info(category_result_folder, category_name, coco_data):
    image_info_list = []

    category_id = 0

    for category in coco_data['categories']:
        if category['name'] == category_name:
            category_id = category['id']
            break

    for file_name in os.listdir(category_result_folder):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            file_path = os.path.join(category_result_folder, file_name)
            image_info = {'address': file_path, 'category': category_name, 'category_id': category_id}
            image_info_list.append(image_info)
    return image_info_list


def find_corresponding_label_ids_file(image_info, ground_folder):
    image_name = os.path.splitext(os.path.basename(image_info['address']))[0]  # 提取图像名称
    label_ids_file = image_name.replace('_leftImg8bit', '_gtFine_labelIds.png')
    label_ids_file_path = os.path.join(ground_folder, label_ids_file)
    return label_ids_file_path

def polygons_to_mask(polygons, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [polygon_np], 1)
    return mask

def calculate_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, np.logical_not(mask2))  # Black and white are reversed in ground and result
    union = np.logical_or(mask1, np.logical_not(mask2))
    iou = np.sum(intersection) / np.sum(union)
    return iou


if __name__ == "__main__":
    result_folder = ".\\coco\\coco_result"
    coco_json_path = '.\\coco\\coco_ground\\coco_ground.json'
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    categories = os.listdir(result_folder)

    ious_per_category = []

    for category in categories:
        ious = []
        category_result_folder = os.path.join(result_folder, category)
        image_info_list = collect_image_info(category_result_folder, category, coco_data)
        if len(image_info_list) == 0:
            print("Category: {:<15} Average IoU: {:<15}".format(category, '0 CLIP dont recognize this category'))
            ious_per_category.append(0)
            continue
        else:
            for image_info in image_info_list:
                image_path = image_info['address']
                image_name = os.path.basename(image_path)
                image_id = 0
                width = 0
                height= 0

                for one_image in coco_data['images']:
                    if one_image['file_name'] == image_name:
                        image_id = one_image['id']
                        width = one_image['width']
                        height = one_image['height']
                        break
                image_annotations = [annotation for annotation in coco_data['annotations'] if
                                     annotation['image_id'] == image_id]
                annotations_with_category = [one_mask for one_mask in image_annotations if one_mask['category_id'] == image_info['category_id']]
                mask_image = np.zeros((height, width, 3), dtype=np.uint8)
                if len(annotations_with_category)>0:
                    one_mask = annotations_with_category[0]
                    segmentation = one_mask['segmentation']
                    ground_mask = polygons_to_mask(segmentation, width, height)

                    result_mask = cv2.imread(image_info['address'],cv2.IMREAD_GRAYSCALE)
                    iou = calculate_iou(ground_mask,result_mask)
                else:
                    iou = 0
                ious.append(iou)

            category_average_iou = np.mean(ious)
            ious_per_category.append(category_average_iou)
            print("Category: {:<15} Average IoU: {:.4f}".format(category, category_average_iou))
    mean_iou = np.mean(ious_per_category)
    print(f"Mean mIOU: {mean_iou}")
