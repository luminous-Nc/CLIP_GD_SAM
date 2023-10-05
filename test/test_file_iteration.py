import os
from PIL import Image
import cv2
import numpy as np

if __name__ == "__main__":
    image_PIL = Image.open("./test_images/road.jpg")
    image = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)
    mask_rgb = image.copy()
    x1, y1, x2, y2 = [100, 200, 500, 600]
    cropped_image = mask_rgb[y1:y2, x1:x2].copy()
    cv2.imshow("hehe", cropped_image)
    cv2.waitKey(0)
    mask_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
