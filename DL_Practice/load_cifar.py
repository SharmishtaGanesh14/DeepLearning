# import torch
# from torchvision import datasets, transforms
#
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset=datasets.CIFAR10(root='./data', download=True, transform=transform)
# test_dataset=datasets.CIFAR10(root='./data', download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
#
# images, labels = next(iter(train_loader))
# print("Images shape:", images.shape)  # [batch_size, channels, height, width]
# print("Labels shape:", labels.shape)
# print("Pixel value range: ", (images.min().item(), images.max().item()))

import numpy as np
import pickle
import os

# Path to CIFAR-10 batches
data_dir = './data/cifar-10-batches-py'

# Function to load a single batch
def load_batch(batch_file):
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']  # shape (10000, 3072)
        labels = batch[b'labels']
        # reshape data to (10000, 3, 32, 32)
        data = data.reshape(-1, 3, 32, 32)
        return data, labels

# Load all training batches
X_train = []
y_train = []
for i in range(1, 6):
    data, labels = load_batch(os.path.join(data_dir, f'data_batch_{i}'))
    X_train.append(data)
    y_train.extend(labels)

X_train = np.concatenate(X_train, axis=0)  # (50000, 3, 32, 32)
y_train = np.array(y_train)                # (50000,)

# Load test batch
X_test, y_test = load_batch(os.path.join(data_dir, 'test_batch'))
y_test = np.array(y_test)

print("Pixel range:", X_train.min(), "to", X_train.max())

# Manual normalization to [0,1]
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Pixel range:", X_train.min(), "to", X_train.max())
