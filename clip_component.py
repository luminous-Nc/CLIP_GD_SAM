import os

import cv2
import numpy as np
import torch
import clip


def get_clip_model(word_list):
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(clip.available_models())
    model, preprocess = clip.load("ViT-B/32", device=device)

    print("clip device: ", device)

    text_inputs = word_list
    text_tokens = clip.tokenize(text_inputs).to(device)

    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return model, preprocess, device, text_features


def get_word_from_clip(image_PIL, model, preprocess, device, word_list, text_features):
    image_input = preprocess(image_PIL).unsqueeze(0).to(device)

    text_inputs = word_list

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    results = []

    for i in range(similarity.shape[0]):
        similarity_num = (100.0 * similarity[i][0])
        text_input = text_inputs[i]
        results.append({"text_input": text_input, "similarity": similarity_num})
        # print(similarity_num)

    results.sort(key=lambda x: x["similarity"], reverse=True)

    detect_word = results[0]["text_input"]
    return detect_word


def get_word_id(word, word_list):
    try:
        index = word_list.index(word)
        return index
    except ValueError:
        return -1


def resend_clip_to_check(word, boundingbox_list, mask_list, image_PIL, model, preprocess, device, word_list,
                         text_features):
    # print(f"Old word: {word}")
    process_boundingbox_list = []
    process_mask_list = []
    image = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)
    for i, mask in enumerate(mask_list):
        mask_rgb = image.copy()
        mask_rgb[mask == 0] = [255, 255, 255]
        resend_word = get_word_from_clip(image_PIL, model, preprocess, device, word_list, text_features)
        # print(f"New word: {word}")
        if word == resend_word:
            process_mask_list.append(mask)
            process_boundingbox_list.append(boundingbox_list[i])
        else:
            print(f'Not equal Old:{word} New: {resend_word}')
            # Discard this mask and bounding box

    return process_boundingbox_list, process_mask_list
