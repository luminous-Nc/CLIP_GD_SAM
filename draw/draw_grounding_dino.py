import os
import datetime

import cv2
from PIL import Image, ImageFile

word_list = ["apple", "banana", "bread", "carrot", "coffee", "corn", "cucumber", "egg", "ice cream",
             "lemon", "milk", "noodles", "peach", "pineapple", "potato", "rice", "sausage", "shrimp", "strawberry",
             "tomato"
             ]

from grounding_module import get_bb_from_grounding_dino

if __name__ == "__main__":
    # threshold = 0.35
    threshold = 0.40
    # threshold = 0.65
    orginal_folder = "./dataset/food_103/images"
    detect_folder = "./dataset/food_103/result_detect"
    ground_info_list = []
    orginal_info_list = []

    word_array = " . ".join(word_list)
    print(word_array)

    for file_name in os.listdir(orginal_folder):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(orginal_folder, file_name)
            orginal_info_list.append(file_path)

    for file_path in orginal_info_list:
        print(f"Begin to detect {file_path}")
        image_PIL = Image.open(file_path)
        boundingbox_list, annotated_bb_image = get_bb_from_grounding_dino(image_PIL, word_array)
        image_file_name = os.path.basename(file_path)
        if len(boundingbox_list) == 0:
            continue
        result_image_path = os.path.join(detect_folder, image_file_name)
        cv2.imwrite(result_image_path, annotated_bb_image.copy())
        print(f"save result to {result_image_path}")
