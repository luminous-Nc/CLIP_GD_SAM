import os
import datetime

import cv2
from PIL import Image, ImageFile

word_list = ["apple", "banana", "bread", "carrot", "coffee", "corn", "cucumber", "egg", "ice cream",
             "lemon", "milk", "noodles", "peach", "pineapple", "potato", "rice", "sausage", "shrimp", "strawberry",
             "tomato"
             ]

word_list = ["rice", "chicken", "apple", "banana"]

from grounding_module import get_bb_from_grounding_dino

if __name__ == "__main__":
    # threshold = 0.35
    threshold = 0.40
    # threshold = 0.65

    ground_info_list = []
    orginal_info_list = []
    file_path = "./dataset/food_103/images/00004402.jpg"
    word_array = " . ".join(word_list)
    print(word_array)

    print(f"Begin to detect {file_path}")
    image_PIL = Image.open(file_path)
    boundingbox_list, annotated_bb_image = get_bb_from_grounding_dino(image_PIL, word_array)
    image_file_name = os.path.basename(file_path)

    cv2.imshow("result", annotated_bb_image)

    cv2.waitKey(0)
