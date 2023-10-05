import os


def get_word_id(word, word_list):
    try:
        index = word_list.index(word)
        return index
    except ValueError:
        return 0  # Return as background


def get_all_images_from_directory(folder_path):
    image_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', 'webp')):
                image_list.append(os.path.join(root, file))
    return image_list


def file_exists_in_directory(directory, base_filename):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            name, ext = os.path.splitext(filename)

            if name == base_filename:
                return True
    return False
