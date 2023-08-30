import time

from PIL import Image
from tqdm import tqdm

from clip_component import get_clip_model, get_word_from_clip
from grounding_component import get_bb_from_grounding_dino
from sam_component import get_mask_predictor, get_mask_from_sam
from save_component import save_data

import os


def process_one_image(image_path, clip_model, clip_preprocess, device, word_list, text_features, mask_predictor,
                      output_path,
                      image_file):
    print(f"process {image_path}")
    image_PIL = Image.open(image_path)

    word = get_word_from_clip(image_PIL, clip_model, clip_preprocess, device, word_list, text_features)
    # print(f"word:{word}")

    boundingbox = get_bb_from_grounding_dino(image_PIL, word)
    if not boundingbox:
        os.remove(image_path)
        print('Empty Detection')
        return
    # print(f"bounding box:{boundingbox}")

    mask = get_mask_from_sam(mask_predictor, image_PIL, boundingbox)
    # print("mask done.")

    save_data(word, boundingbox, mask, output_path, image_file)

    os.remove(image_path)


if __name__ == "__main__":

    word_list = ["person", "bicycle", "car", "motorcycle", "airplane",
                 "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                 "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                 "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                 "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                 "skis", "snowboard", "sports ball", "kite", "baseball bat",
                 "baseball glove", "skateboard", "surfboard", "tennis racket",
                 "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                 "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                 "hot dog", "pizza", "donut", "cake", "chair", "couch",
                 "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                 "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                 "toaster", "sink", "refrigerator", "book", "clock", "vase",
                 "scissors", "teddy bear", "hair drier", "toothbrush"
                 ]

    data_set_path = "../dataset/validation/coco_2017/coco_val"
    output_path = "../dataset/validation/coco_2017/coco_result"

    clip_model, clip_preprocess, device, text_features = get_clip_model(word_list)
    mask_predictor = get_mask_predictor()

    all_image_paths = []
    for root, dirs, files in os.walk(data_set_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                all_image_paths.append(os.path.join(root, file))

    with tqdm(total=len(all_image_paths), desc="Generating CoCo 2017 masks") as pbar:
        for image_path in all_image_paths:
            image_file = os.path.basename(image_path)
            start_time = time.time()
            process_one_image(image_path, clip_model, clip_preprocess, device, word_list, text_features, mask_predictor,
                              output_path,
                              image_file)
            end_time = time.time()
            elapsed_time = end_time - start_time
            pbar.set_postfix({"Time": f"{elapsed_time:.2f} s"})
            pbar.update(1)

    print("Task Finished")
