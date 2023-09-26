task_mappings = {
    "test": {
        "word_list": ['apple', "banana", "bread", "cereal", "chicken", "juice", "lemon", "milk", "orange", "salad"],
        "data_set_path": "./dataset/food_simple/images",
        "output_path": "./dataset/food_simple/results",
        "output_pure_path": "./dataset/food_simple/pure_masks"
    },
    "city": {
        "word_list": ['road', 'sidewalk', 'parking lot', 'rail track', 'person', 'rider', 'car', 'truck', 'bus',
                      'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence',
                      'guard rail', 'bridge', 'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'tree',
                      'terrain', 'sky'],
        "data_set_path": "./dataset/cityscapes/images",
        "output_path": "./dataset/cityscapes/results",
        "output_pure_path": "./dataset/cityscapes/pure_masks"
    },
    "IBL_food": {
        "word_list": ["bacon", "baked beans", "beans and rice", "beef lamb veal", "beverage", "biscuits",
                      "breads", "breadsticks", "breakfast bars", "butter", "cakes", "candy", "cheese",
                      "cheese sandwich", "chili",
                      "chips", "condiments and sauces", "cooked cereal", "cookies", "cornbread", "cottage cheese",
                      "crackers", "cream cheese", "creamers", "creole", "dips", "dumplings", "egg rolls", "eggs",
                      "fast food salads", "fish and seafood",
                      "fish patty", "french fries", "french toast", "fruit", "fruit and vegetables", "gelatin",
                      "gnocchi", "gravy", "ice cream", "lasagna",
                      "luncheon meats", "mayonnaise", "mexican dishes", "olives", "pie", "pretzel", "rolls bagels buns",
                      "sandwich", "shepherd pie", "sour cream", "spaghetti sauce", "sugar", "sweet breads",
                      "sweet breads pastries muffins", "syrups", "syrups lcings",
                      "tortillas tacos shells", "trail and snack mix", "various salads", "vegetables", "waffle",
                      "yogurt"],
        "data_set_path": "./dataset/IBL_food/images",
        "output_path": "./dataset/IBL_food/results",
        "output_pure_path": "./dataset/IBL_food/pure_masks"
    },
    "food": {
        "word_list": ["bacon", "baked beans", "beans and rice", "beef lamb veal", "beverage", "biscuits",
                      "breads", "breadsticks", "breakfast bars", "butter", "cakes", "candy", "cheese",
                      "cheese sandwich", "chili", "chips", "condiments and sauces", "cooked cereal", "cookies", "cornbread", "cottage cheese",
                      "crackers", "cream cheese", "creamers", "creole", "dips", "dumpling", "egg rolls", "eggs",
                      "fast food salads", "fish and seafood",
                      "fish patty", "french fries", "french toast", "fruit", "fruit and vegetables", "gelatin",
                      "gnocchi", "gravy", "ice cream", "lasagna",
                      "luncheon meats", "mayonnaise", "mexican dishes", "olives", "pie", "pretzel", "rolls bagels buns",
                      "sandwich", "shepherd pie", "sour cream", "spaghetti sauce", "sugar", "sweet breads",
                      "sweet breads pastries muffins", "syrups", "syrups lcings",
                      "tortillas tacos shells", "trail and snack mix", "various salads", "vegetables", "waffle",
                      "yogurt"],
        "data_set_path": "./dataset/food/images",
        "output_path": "./dataset/food/results",
        "output_pure_path": "./dataset/food/pure_masks"
    },

    "coco": {
        "word_list": ["person", "bicycle", "car", "motorcycle", "airplane",
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
                      "scissors", "teddy bear", "hair drier", "toothbrush"],
        "data_set_path": "dataset/coco_2017/images",
        "output_path": "dataset/coco_2017/results",
    },
    "coca": {
        "word_list": ['Accordion', 'alarm clock', 'avocado', 'backpack', 'baseball', 'beer bottle', 'belt',
                      'binoculars', 'boots', 'butterfly', 'calculator', 'camel', 'camera', 'candle', 'chopsticks',
                      'clover', 'dice',
                      'dolphin', 'doughnut', 'dumbbell', 'eggplant', 'faucet', 'fishing rod', 'frisbee', 'gift box',
                      'glasses', 'globe', 'glove', 'guitar', 'hammer', 'hammock', 'handbag', 'harp', 'hat', 'headphone',
                      'helicopter', 'high heels', 'hourglass', 'ice cream', 'key', 'lollipop', 'macaroon', 'microphone',
                      'minions', 'moon', 'persimmon', 'pigeon', 'pillow', 'pine cone', 'pineapple', 'pocket watch',
                      'poker',
                      'potato', 'pumpkin', 'rabbit', 'rocking horse', 'roller-skating', 'rolling pin', 'soap bubble',
                      'squirrel', 'stethoscope', 'sticky note', 'stool', 'strawberry', 'sunflower', 'tablet',
                      'teddy bear',
                      'thermometer', 'tomato', 'towel', 'toy car', 'typewriter', 'UAV', 'violin', 'waffles',
                      'watering can',
                      'watermelon', 'wheelchair', 'whisk', 'Yellow duck'],
        "data_set_path": "./dataset/coca/images",
        "output_path": "./dataset/coca/results",

    },
    "voc": {
        "word_list": ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus",
                      "car", "motorbike", "train", "bottle", "chair", "dining table", "potted plant", "sofa",
                      "TV/monitor"],
        "data_set_path": "./dataset/validation/pascal_voc2012/voc2012_val",
        "output_path": "./dataset/validation/pascal_voc2012/voc2012_result"
    },
    "fashionpedia": {
        "word_list": ['shirt', 'blouse', 'top', 't-shirt', 'sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest',
                      'pants',
                      'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband',
                      'head covering',
                      'hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights', 'stockings', 'sock',
                      "shoe",
                      'bag', 'wallet', "scarf", "umbrella", "hood", "collar", "lapel", "epaulette", "sleeve",
                      'pocket', "neckline", "buckle", "zipper", "applique", "bead", "bow", "flower", "fringe", "ribbon",
                      "rivet", "ruffle", "sequin", "tassel"],
        "data_set_path": "./dataset/fashionpedia/images",
        "output_path": "./dataset/fashionpedia/results"
    }

}
