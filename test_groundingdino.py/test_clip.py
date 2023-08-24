from clip_component import get_clip_model, get_word_from_clip
from PIL import Image

if __name__ == "__main__":
    clip_model, clip_preprocess, device = get_clip_model()
    word_list = ['road', 'sidewalk', 'parking lot', 'rail track', 'person', 'rider', 'car', 'truck', 'bus', 'on rails',
                 'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence', 'guard rail', 'bridge',
                 'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'tree', 'terrain', 'sky']
    image_PIL = Image.open("road.jpg")
    word = get_word_from_clip(image_PIL, clip_model, clip_preprocess, device, word_list)
    print(word)