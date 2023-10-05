import os

import numpy as np

from clip_module import get_clip_model, get_word_from_clip
from PIL import Image

if __name__ == "__main__":
    word_list = ["bacon", "baked bean", "beans and rice", "beef lamb veal", "beverage", "biscuit",
                 "bread", "breadstick", "breakfast bar", "butter", "cake", "candy", "cheese",
                 "cheese sandwich", "chili",
                 "chip", "condiments and sauce", "cooked cereal", "cookie", "cornbread", "cottage cheese",
                 "cracker", "cream cheese", "creamer", "creole", "dip", "dumpling", "egg roll", "egg",
                 "fast food salad", "fish and seafood",
                 "fish patty", "french fries", "french toast", "fruit", "fruit and vegetable", "gelatin",
                 "gnocchi", "gravy", "ice cream", "lasagna",
                 "luncheon meat", "mayonnaise", "mexican dishes", "olive", "pie", "pretzel", "rolls bagels bun",
                 "sandwich", "shepherd pie", "sour cream", "spaghetti sauce", "sugar", "sweet bread",
                 "sweet bread pastry muffin", "syrups", "syrups lcing",
                 "tortilla taco shell", "trail and snack mix", "various salad", "vegetable", "waffle",
                 "yogurt"]

    ground_folder = ".\\..\\dataset\\IBL_food\\images"
    # result_folder = ".\\..\\dataset\\IBL_food\\results"
    result_folder = ".\\..\\dataset\\IBL_food\\results_clip"

    categories = os.listdir(ground_folder)
    accuracy_per_category = []

    for category in categories:
        category_ground_path = os.path.join(ground_folder, category)
        category_result_path = os.path.join(result_folder, category.lower())

        ground_images = os.listdir(category_ground_path)
        result_images = os.listdir(category_result_path)

        accuracys = []

        for image_name in ground_images:
            ground_mask_path = os.path.join(category_ground_path, image_name)
            corresponding_result_name = image_name[:-4] + '.jpg'
            result_mask_path = os.path.join(category_result_path, corresponding_result_name)

            if not os.path.exists(result_mask_path):
                accuracy = 0
                accuracys.append(accuracy)
            else:
                accuracy = 100
                accuracys.append(accuracy)

        category_average_iou = np.mean(accuracys)
        accuracy_per_category.append(category_average_iou)
        print("Category: {:<15} Average Accuracy: {:.4f}%".format(category, category_average_iou))

    mean_accuracy = np.mean(accuracy_per_category)
    print(f"Mean mAccuracy: {mean_accuracy}")
