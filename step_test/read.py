from PIL import Image
import cv2
import numpy as np
if __name__ == "__main__":
    image_PIL = Image.open("milk.jpg")

    label_ids = np.array(image_PIL)
    image_CV = cv2.imread("milk.jpg")
    print(label_ids)