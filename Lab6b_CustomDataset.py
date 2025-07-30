import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)  # Correct way to load image from file path
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Use your actual CSV file path and image directory here
annotations_file = './data/labels.csv'
img_dir = './data/images'

transform = ToTensor()  # If your images are not already in tensor form

training_data = CustomImageDataset(
    annotations_file=annotations_file,
    img_dir=img_dir,
    transform=transform
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Display image and label
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze().permute(1, 2, 0)  # Convert from CxHxW to HxWxC
label = train_labels[0]

plt.imshow(img)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
