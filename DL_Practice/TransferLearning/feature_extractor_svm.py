import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torchmetrics.functional import confusion_matrix
from torchvision import models, datasets, transforms
import numpy as np

batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed=42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

def load_dataset():
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])
    train_dataset = datasets.CIFAR10(train=True, download=True, transform=transform,root='../data')
    test_dataset = datasets.CIFAR10(train=False, download=True, transform=transform,root='../data')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def feature_extractor(model,loader):
    features = []
    targets = []
    model.eval()
    with torch.no_grad():
        for data,target in loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    return np.concatenate(features),np.concatenate(targets)

model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
weights=models.ResNet18_Weights.IMAGENET1K_V1
print(weights.transforms())
for param in model.parameters():
    param.requires_grad=False

Feat_Extract=nn.Sequential(*list(model.children())[:-1])
Feat_Extract.to(device)
train_loader, test_loader = load_dataset()
X_train,y_train = feature_extractor(Feat_Extract,train_loader)
X_test,y_test = feature_extractor(Feat_Extract,test_loader)

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

svc = SVC(kernel="rbf", gamma="auto", random_state=42)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)