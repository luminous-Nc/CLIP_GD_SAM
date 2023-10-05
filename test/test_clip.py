from clip_module import CLIPModel
from PIL import Image

if __name__ == "__main__":
    word_list = ['apple', "banana", "bread", "cereal", "chicken", "juice", "lemon", "milk", "orange", "salad",
                 "breakfast bars", "chip", "cake", "barcode"]
    clip_model = CLIPModel(word_list)

    image_PIL = Image.open("test_images/0016000502666.jpg")
    word = clip_model.get_word_list_from_clip(image_PIL, 4)
    print(word)
