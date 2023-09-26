from clip_component import get_clip_model, get_word_from_clip
from PIL import Image

if __name__ == "__main__":
    word_list = ["bacon", "baked bean", "beans and rice", "beef lamb veal", "beverage", "biscuit",
                 "bread", "breadstick", "breakfast bars", "butter", "cakes", "candy", "cheese",
                 "cheese sandwich", "chili",
                 "chip", "condiments and sauce", "cooked cereal", "cookie", "cornbread", "cottage cheese",
                 "cracker", "cream cheese", "creamer", "creole", "dip", "dumpling", "egg roll", "egg",
                 "fast food salad", "fish and seafood",
                 "fish patty", "french fries", "french toast", "fruit", "fruit and vegetable", "gelatin",
                 "gnocchi", "gravy", "ice cream", "lasagna",
                 "luncheon meat", "mayonnaise", "mexican dishes", "olive", "pie", "pretzel", "rolls bagels bun",
                 "sandwich", "shepherd pie", "sour cream", "spaghetti sauce", "sugar", "sweet bread",
                 "sweet bread pastry muffin", "syrups", "syrups lcing",
                 "tortilla taco shell", "trail and snack mix", "various salad", "vegetable", "waffle",
                 "yogurt"]
    # word_list = ['apple', "banana", "bread", "cereal", "chicken", "juice", "lemon", "milk", "orange", "salad",
    #                   "breakfast bars", "chip", "cake", ]
    clip_model, clip_preprocess, device, text_features = get_clip_model(word_list)
    image_PIL = Image.open("041708103.jpg")
    word = get_word_from_clip(image_PIL, clip_model, clip_preprocess, device, word_list, text_features)
    print(word)
