import os


def prepare_dataset_and_create_dir(dataset_path, output_path):
    # Get all sub-folder's name within a folder
    category_folders = [folder_name for folder_name in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, folder_name))]

    for category_folder in category_folders:
        result_category_path = os.path.join(output_path, category_folder)
        os.makedirs(result_category_path, exist_ok=True)

    return category_folders


def prepare_categories_and_create_dir(output_path):
    category_folders = ['road', 'sidewalk', 'parking lot', 'rail track', 'person', 'rider', 'car', 'truck', 'bus',
                        'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence',
                        'guard rail', 'bridge', 'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'tree',
                        'terrain', 'sky']
    for category_folder in category_folders:
        result_category_path = os.path.join(output_path, category_folder)
        os.makedirs(result_category_path, exist_ok=True)


if __name__ == "__main__":
    dataset_path = "dataset/cityscapes/city"
    output_path = "dataset/cityscapes/city_result"
    prepare_categories_and_create_dir(output_path)
