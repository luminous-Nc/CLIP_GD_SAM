import os

import numpy as np

from clip_module import get_clip_model, get_word_from_clip
from PIL import Image
import pandas as pd

if __name__ == "__main__":
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

    ground_folder = ".\\..\\dataset\\food_103\\ground_truths"
    # result_folder = ".\\..\\dataset\\food_103\\results"
    result_folder = ".\\..\\dataset\\food_103\\results_clip"

    categories = os.listdir(result_folder)
    accuracy_per_category = []
    categories_excel = []
    accuracy_excel = []

    for category in categories:
        accuracys = []

        category_result_path = os.path.join(result_folder, category)
        result_images = os.listdir(category_result_path)
        category_id = word_list.index(category)

        if len(result_images) == 0:
            categories_excel.append(category)
            accuracy_excel.append('CLIP dont recognize')
            continue
        else:
            for image_name in result_images:
                ground_truth_path = os.path.join(ground_folder, image_name[:-4] + '.png')

                ground_truth_image = Image.open(ground_truth_path)
                ground_truth_array = np.array(ground_truth_image)

                unique_ids = np.unique(ground_truth_array)

                object_strings = np.array([word_list[category_index] for category_index in unique_ids])

                is_target_category_present = np.isin(category_id, unique_ids)

                if is_target_category_present:
                    accuracy = 100
                    accuracys.append(accuracy)
                else:
                    accuracy = 0
                    # print(f"{image_name} whose category {category} is not present on ground truth which has:{object_strings}")
                    accuracys.append(accuracy)

        category_average_accuracy = np.mean(accuracys)
        accuracy_per_category.append(category_average_accuracy)
        categories_excel.append(category)
        accuracy_excel.append(category_average_accuracy)

        print("Category: {:<15} Average Accuracy: {:.4f}".format(category, category_average_accuracy))

    mean_accuracy = np.mean(accuracy_per_category)
    print(f"Total {len(accuracy_per_category)} categories")
    print(f"Mean mAccuracy: {mean_accuracy}")

    df = pd.DataFrame({'category': categories_excel, "accuracy": accuracy_excel})
    df.to_excel("after_clip_accuracy.xlsx")
