from clip_component import get_clip_model, get_word_from_clip
from PIL import Image
import os
if __name__ == "__main__":
    clip_model, clip_preprocess, device = get_clip_model()
    word_list = ['shirt', 'blouse', 'top', 't-shirt', 'sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants',
                 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband', 'head covering',
                 'hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights', 'stockings', 'sock', "shoe",
                 'bag', 'wallet', "scarf", "umbrella", "hood", "collar", "lapel", "epaulette", "sleeve",
                 'pocket', "neckline", "buckle", "zipper", "applique", "bead", "bow", "flower", "fringe", "ribbon",
                 "rivet", "ruffle", "sequin", "tassel"]
    data_set_path = "../dataset/fashionpedia/test"

    # Get all files list
    all_image_paths = []
    for root, dirs, files in os.walk(data_set_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                all_image_paths.append(os.path.join(root, file))
    # print(all_image_paths)
    #
    #
    for image_path in all_image_paths:
        image_file_name = os.path.basename(image_path)
        image_PIL = Image.open(image_path)
        # print(image_file_name)
        word = get_word_from_clip(image_PIL, clip_model, clip_preprocess, device, word_list)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        result_txt_path = os.path.join(data_set_path, image_name) + "_"+ word + ".txt"
        print(result_txt_path)
        with open(result_txt_path, "w") as txt_file:
            txt_file.write(str(word))
    print("Task Finished")