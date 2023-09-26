from clip_component import get_clip_model, get_word_from_clip
from PIL import Image

if __name__ == "__main__":
    word_list = ["bacon", "baked beans", "beans and rice", "beef lamb veal", "beverage", "biscuits",
                 "breads", "breadsticks", "breakfast bars", "butter", "cakes", "candy", "cheese", "cheese sandwich",
                 "chili","fruit","orange"
                 "chips", "condiments and sauces", "cooked cereal", "cookies", "cornbread", "cottage cheese",
                 "crackers", "cream cheese", "creamers", "creole", "dips", "dumpling", "egg rolls", "eggs",
                 "fast food salads", "fish and seafood",
                 "fish patty", "french fries", "french toast", "fruit", "fruit and vegetables", "gelatin", "gnocchi",
                 "gravy", "ice cream", "lasagna",
                 "luncheon meats", "mayonnaise", "mexican dishes", "olives", "pie", "pretzel", "rolls bagels buns",
                 "sandwich", "shepherd pie", "sour cream", "spaghetti sauce", "sugar", "sweet breads",
                 "sweet breads pastries muffins", "syrups", "syrups lcings",
                 "tortillas tacos shells", "trail and snack mix", "various salads", "vegetables", "waffle", "yogurt"]
    clip_model, clip_preprocess, device, text_features = get_clip_model(word_list)
    image_PIL = Image.open("hum.jpg")
    word = get_word_from_clip(image_PIL, clip_model, clip_preprocess, device, word_list, text_features)
    print(word)
