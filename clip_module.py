import os

import cv2
import numpy as np
import torch
import clip
from PIL import Image


class CLIPModel:
    def __init__(self, word_list, fine_tune=False):
        self.word_list = word_list
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        if fine_tune:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
            checkpoint = torch.load("fine_tune_CLIP.pt")
            self.model.load_state_dict(checkpoint['model_state_dict'])
        text_inputs = word_list
        text_tokens = clip.tokenize(text_inputs).to(self.device)
        self.text_features = self.model.encode_text(text_tokens).float()
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def get_a_word(self, image_PIL):
        image_input = self.preprocess(image_PIL).unsqueeze(0).to(self.device)
        text_inputs = self.word_list
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            similarity = self.text_features.cpu().numpy() @ image_features.cpu().numpy().T
        results = []
        for i in range(similarity.shape[0]):
            similarity_num = (100.0 * similarity[i][0])
            text_input = text_inputs[i]
            # similarity_num_formatted = "{:.4f}".format(similarity_num)
            results.append({"text_input": text_input, "similarity": similarity_num})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        detect_word = results[0]["text_input"]
        return detect_word

    def get_word_list_from_clip(self, image_PIL, select_number):
        image_input = self.preprocess(image_PIL).unsqueeze(0).to(self.device)
        text_inputs = self.word_list

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            similarity = self.text_features.cpu().numpy() @ image_features.cpu().numpy().T

        results = []

        for i in range(similarity.shape[0]):
            similarity_num = (100.0 * similarity[i][0])
            text_input = text_inputs[i]
            results.append({"text_input": text_input, "similarity": similarity_num})

        if len(results) == 0:
            return results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        selected_results = [result["text_input"] for result in results[:int(select_number)]]

        return selected_results

    def recheck_from_sam(self, boundingbox_list, mask_list, image_PIL):
        process_boundingbox_list = []
        process_mask_list = []
        image = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)
        for i, mask in enumerate(mask_list):
            boundingbox = boundingbox_list[i]
            word = boundingbox["word"]
            mask_rgb = image.copy()
            mask_rgb[mask == 0] = [255, 255, 255]  # set background to pure white
            x1, y1, x2, y2 = boundingbox["bb_xyxy"]
            cropped_image = mask_rgb[y1:y2, x1:x2].copy()
            mask_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            resend_word = self.get_a_word(mask_pil)

            cv2.imshow(f"old {word} new {resend_word}", cropped_image)
            cv2.waitKey(0)
            if word == resend_word:
                process_mask_list.append(mask)
                process_boundingbox_list.append(boundingbox_list[i])
            else:
                print(f'Not equal Old:{word} New: {resend_word}')
                # Discard this mask and bounding box

        return process_boundingbox_list, process_mask_list
