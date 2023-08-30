from grounding_component import get_bb_from_grounding_dino
from PIL import Image

if __name__ == "__main__":
    image_PIL = Image.open("road.jpg")
    word = "road"
    boundingbox = get_bb_from_grounding_dino(image_PIL, word)
    print(boundingbox)
