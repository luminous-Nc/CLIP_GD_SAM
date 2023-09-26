import os
from pipeline_config import task_mappings

def prepare_categories_and_create_dir(task_args):
    task_info = task_mappings.get(task_args.task)
    if task_info:
        word_list = task_info["word_list"]
        output_path = task_info["output_path"]
        if not os.listdir(output_path):
            print(f"Task : {task_args.task} has empty result folders, create category folders")
            for category_folder in word_list:
                result_category_path = os.path.join(output_path, category_folder)
                os.makedirs(result_category_path, exist_ok=True)
        else:
            print(f"Task : {task_args.task} already has result folders ")
    else:
        print(f"Unknown taskname: {task_args.task}")

if __name__ == "__main__":

    output_path = "./dataset/IBL_food/IBL results"
    prepare_categories_and_create_dir(output_path)
