import os
from pipeline_config import task_mappings


def prepare_categories_and_create_dir(task_args):
    task_info = task_mappings.get(task_args.task)
    if task_info:
        word_list = task_info["word_list"]
        output_path = task_info["output_path"]
        output_pure_path = task_info["output_pure_path"]
        if task_args.result:
            if not os.listdir(output_path):
                print(f"Task : {task_args.task} has empty result folders, create category folders")
                for category_folder in word_list:
                    result_category_path = os.path.join(output_path, category_folder)
                    print(f"create folder: {category_folder}")
                    os.makedirs(result_category_path, exist_ok=True)
            else:
                print(f"Task : {task_args.task} already has result folders ")
        if task_args.pure_mask or task_args.bounding_box:
            if not os.listdir(output_pure_path):
                print(f"Task : {task_args.task} has empty pure mask folders, create category folders")
                for category_folder in word_list:
                    result_category_path = os.path.join(output_pure_path, category_folder)
                    print(f"create folder: {category_folder}")
                    os.makedirs(result_category_path, exist_ok=True)
            else:
                print(f"Task : {task_args.task} already has pure mask folders ")
    else:
        print(f"Unknown task name: {task_args.task}")
