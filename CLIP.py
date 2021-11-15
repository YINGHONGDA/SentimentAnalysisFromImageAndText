import torch
import clip
from PIL import Image
import numpy as np

with open('/home/yinghongda/DataSet/VSO.txt', 'r') as f:
    line = f.readlines()
    labels = []
    for i in line:
        i = i.strip()
        k = i.split(' ')[0]
        labels.append(k)

def get_imageContent(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image  = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=1).cpu().numpy()

    content  = np.where(probs == np.max(probs))[1]

    return labels[content.tolist()[0]]

if __name__ == '__main__':
    image = get_imageContent("/home/yinghongda/DataSet/predict/amazing sunset.jpg")
    print(image)





