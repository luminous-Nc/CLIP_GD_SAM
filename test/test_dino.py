from grounding_module import get_bb_from_grounding_dino
from PIL import Image

if __name__ == "__main__":
    image_PIL = Image.open("test_images/road.jpg")
    word = "road"
    boundingbox = get_bb_from_grounding_dino(image_PIL, word)
    print(boundingbox)
