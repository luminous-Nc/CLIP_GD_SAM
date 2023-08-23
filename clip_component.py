import torch
from PIL import Image
import clip


def get_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    print("clip device: ", device)

    return model, preprocess, device
def get_word_from_clip(image_PIL, model, preprocess, device , word_list):

    image_input = preprocess(image_PIL).unsqueeze(0).to(device)

    text_inputs = word_list
    text_tokens = clip.tokenize(text_inputs).to(device)

    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)


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

    # Print the caption for each text input along with their similarity scores

    # for result in results:
    #     print(f"Text input: {result['text_input']}, Similarity: {result['similarity']:.2f}")
    #     if result['similarity'] >= similarity_threshold:
    #         detect_food += " " + result['text_input'] + ", "

    detect_word = results[0]["text_input"]
    return detect_word
