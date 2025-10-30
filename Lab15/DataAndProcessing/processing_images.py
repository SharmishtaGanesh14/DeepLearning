import torch
from PIL import Image
from torch import nn
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
from torchvision.models import ResNet50_Weights


def main():
    # load ground truth csv
    df = pd.read_csv("../../data/Flick8K/captions.txt", sep=',')

    # load default pretrained weights
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # remove the final FC layer
    modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*modules)
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # imageNet values
                             std=[0.229, 0.224, 0.225])
    ])

    def extract_features(image_path, model, transform):
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model(image)  # [1, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [1, 2048]
        return features.squeeze(0).numpy()

    features = []
    for img in df["image"].unique():
        img_path = f"Images/{img}"
        if os.path.exists(img_path):
            feat = extract_features(img_path, resnet, transform)
            row={
                "image": img,
                "features": feat
            }
            features.append(row)

    images_fc = pd.DataFrame(features)
    images_fc.to_pickle("features_to_rnn.pkl")
    print(f"Saved features for {len(images_fc)} images to features_to_rnn.pkl")

if __name__ == "__main__":
    main()