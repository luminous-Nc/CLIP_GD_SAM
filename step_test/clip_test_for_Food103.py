import os

import numpy as np

from clip_component import get_clip_model, get_word_from_clip
from PIL import Image

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

    ground_folder = ".\\..\\dataset\\food_103\\images"
    result_folder = ".\\..\\dataset\\food_103\\results"

    categories = os.listdir(result_folder)
    accuracy_per_category = []

    for category in categories:
        accuracys = []

        category_result_path = os.path.join(result_folder, category)
        result_images = os.listdir(category_result_path)
        category_id = word_list.index(category)

        if len(result_images)== 0:
            accuracys = [100]
        else:
            for image_name in result_images:
                ground_truth_path = os.path.join(ground_folder,image_name[:-4] + '.jpg')

                ground_truth_image = Image.open(ground_truth_path)
                ground_truth_array = np.array(ground_truth_image)

                is_target_category_present = np.any(ground_truth_array == category_id)

                if is_target_category_present:
                    accuracy = 100
                    accuracys.append(accuracy)
                else:
                    accuracy = 0
                    accuracys.append(accuracy)

        category_average_accuracy = np.mean(accuracys)
        accuracy_per_category.append(category_average_accuracy)
        print("Category: {:<15} Average Accuracy: {:.4f}".format(category, category_average_accuracy))

    mean_accuracy = np.mean(accuracy_per_category)
    print(f"Mean mAccuracy: {mean_accuracy}")
