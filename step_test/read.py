from PIL import Image
import cv2
import numpy as np
if __name__ == "__main__":
    image_PIL = Image.open("00006948.png")

    label_ids = np.array(image_PIL)
    image_CV = cv2.imread("00006948.png")
    print(label_ids)