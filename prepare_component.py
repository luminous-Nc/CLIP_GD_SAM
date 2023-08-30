import os


def prepare_dataset_and_create_dir(dataset_path, output_path):
    category_folders = [folder_name for folder_name in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, folder_name))]

    for category_folder in category_folders:
        result_category_path = os.path.join(output_path, category_folder)
        os.makedirs(result_category_path, exist_ok=True)

    return category_folders


def prepare_categories_and_create_dir(output_path):
    category_folders = ['road', 'sidewalk', 'parking lot', 'rail track', 'person', 'rider', 'car', 'truck', 'bus',
                        'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence',
                        'guard rail', 'bridge', 'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'tree',
                        'terrain', 'sky']

    # category_folders = ['Accordion', 'alarm clock', 'avocado', 'backpack', 'baseball', 'beer bottle', 'belt', 'binoculars',
    #              'boots', 'butterfly', 'calculator', 'camel', 'camera', 'candle', 'chopsticks', 'clover', 'dice',
    #              'dolphin', 'doughnut', 'dumbbell', 'eggplant', 'faucet', 'fishing rod', 'frisbee', 'gift box',
    #              'glasses', 'globe', 'glove', 'guitar', 'hammer', 'hammock', 'handbag', 'harp', 'hat', 'headphone',
    #              'helicopter', 'high heels', 'hourglass', 'ice cream', 'key', 'lollipop', 'macaroon', 'microphone',
    #              'minions', 'moon', 'persimmon', 'pigeon', 'pillow', 'pine cone', 'pineapple', 'pocket watch', 'poker',
    #              'potato', 'pumpkin', 'rabbit', 'rocking horse', 'roller-skating', 'rolling pin', 'soap bubble',
    #              'squirrel', 'stethoscope', 'sticky note', 'stool', 'strawberry', 'sunflower', 'tablet', 'teddy bear',
    #              'thermometer', 'tomato', 'towel', 'toy car', 'typewriter', 'UAV', 'violin', 'waffles', 'watering can',
    #              'watermelon', 'wheelchair', 'whisk', 'Yellow duck']

    # category_folders = ["person", "bicycle", "car", "motorcycle", "airplane",
    #                     "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    #                     "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    #                     "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    #                     "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    #                     "skis", "snowboard", "sports ball", "kite", "baseball bat",
    #                     "baseball glove", "skateboard", "surfboard", "tennis racket",
    #                     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    #                     "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    #                     "hot dog", "pizza", "donut", "cake", "chair", "couch",
    #                     "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    #                     "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    #                     "toaster", "sink", "refrigerator", "book", "clock", "vase",
    #                     "scissors", "teddy bear", "hair drier", "toothbrush"
    #                     ]

    category_folders = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus",
                        "car", "motorbike", "train", "bottle", "chair", "dining table", "potted plant", "sofa",
                        "TV or monitor/monitor"]

    for category_folder in category_folders:
        result_category_path = os.path.join(output_path, category_folder)
        os.makedirs(result_category_path, exist_ok=True)


if __name__ == "__main__":
    dataset_path = "dataset/validation/pascal_voc2012/voc2012_val"
    output_path = "dataset/validation/pascal_voc2012/voc2012_result"
    prepare_categories_and_create_dir(output_path)
