from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn
import torch
import numpy as np

# Config
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '../data'
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model: ResNet18 feature extractor
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in resnet.parameters():
    param.requires_grad = False  # freeze backbone
Feature_Extractor = nn.Sequential(*list(resnet.children())[:-1])  # remove final FC
Feature_Extractor = Feature_Extractor.to(DEVICE)

# Data Loading
def load_data():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_data = datasets.CIFAR10(root=DATA_PATH, train=True,
                                  transform=transform, download=True)
    test_data = datasets.CIFAR10(root=DATA_PATH, train=False,
                                 transform=transform, download=True)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

# Feature extraction
def extract_features(model, loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            output = model(data)           # (B, 512, 1, 1)
            output = output.view(output.size(0), -1)  # flatten -> (B, 512)
            features.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

# Main pipeline
train_loader, test_loader = load_data()
X_train, y_train = extract_features(Feature_Extractor, train_loader)
X_test, y_test = extract_features(Feature_Extractor, test_loader)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
clf = SVC(kernel='rbf', C=1, random_state=RANDOM_SEED)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, pred))
