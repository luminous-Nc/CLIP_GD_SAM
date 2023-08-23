import os


def prepare_dataset_and_create_dir(dataset_path,output_path):
    # Get all sub-folder's name within a folder
    category_folders = [folder_name for folder_name in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, folder_name))]

    for category_folder in category_folders:
        result_category_path = os.path.join(output_path, category_folder)
        os.makedirs(result_category_path, exist_ok=True)

    return category_folders
