import os
import datetime

import cv2
import numpy as np
import pandas as pd
import openpyxl

from clip_module import get_clip_model, get_word_from_clip
from PIL import Image

category_valid_id = [0, 25, 29, 58, 84, 13, 59, 82, 24, 8, 37, 15, 65, 36, 40, 70, 66, 49, 56, 30, 73]

if __name__ == "__main__":
    ground_truth_mask = Image.open("../dataset/food_103/ground_truths/00005990.png")
    result_mask = Image.open("../dataset/food_103/pure_masks_simple - 0.5/00005990.png")
    width, height = result_mask.size
    image_file_name = os.path.basename("\\dataset\\food_103\\pure_masks_simple - 0.5\\00005990.png")
    ppt_image_path = os.path.join("../dataset/food_103/ppt_result", image_file_name)

    result_image = Image.new("RGB", (width, height))

    for y in range(height):
        for x in range(width):
            mask_pixel = result_mask.getpixel((x, y))[0]
            gt_pixel = ground_truth_mask.getpixel((x, y))
            if mask_pixel == gt_pixel and mask_pixel == 0 and gt_pixel == 0:  # If both are background
                result_image.putpixel((x, y), (0, 0, 0))  # Black
            elif mask_pixel in category_valid_id and gt_pixel in category_valid_id:
                if gt_pixel == mask_pixel:
                    result_image.putpixel((x, y),
                                          (mask_pixel, mask_pixel, mask_pixel))  # Both are same category correct
                elif mask_pixel == 0 and gt_pixel != 0:  # mask not segment but ground truth has
                    result_image.putpixel((x, y), (255, 0, 0))  # Red
                elif mask_pixel != 0 and gt_pixel != 0:  # Incorrect category
                    result_image.putpixel((x, y), (255, 255, 0))  # Yellow
                elif mask_pixel != 0 and gt_pixel == 0:  # mask segment but don't in ground
                    result_image.putpixel((x, y), (0, 0, 255))  # Blue

    result_image.save(ppt_image_path)
