from clip_component import get_clip_model, get_word_from_clip
from PIL import Image

if __name__ == "__main__":
    clip_model, clip_preprocess, device = get_clip_model()
    word_list = ['shirt', 'blouse', 'top', 't-shirt', 'sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants',
                 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband', 'head covering',
                 'hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights', 'stockings', 'sock', "shoe",
                 'bag', 'wallet', "scarf", "umbrella", "hood", "collar", "lapel", "epaulette", "sleeve",
                 'pocket', "neckline", "buckle", "zipper", "applique", "bead", "bow", "flower", "fringe", "ribbon",
                 "rivet", "ruffle", "sequin", "tassel"]
    image_PIL = Image.open("road.jpg")
    word = get_word_from_clip(image_PIL, clip_model, clip_preprocess, device, word_list)
    print(word)
