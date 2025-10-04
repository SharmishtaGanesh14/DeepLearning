import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torchvision import datasets, transforms, models

seed=41
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Feature_Extractor(model,loader):
    model.eval()
    features = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output=output.view(output.size(0),-1)
            features.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

    return np.concatenate(features), np.concatenate(targets)

def main():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    model = nn.Sequential(*list(model.children())[:-1])

    weights=models.ResNet18_Weights.IMAGENET1K_V1
    print(weights.transforms())

    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                    ])
    train_dataset = datasets.CIFAR10(root='../data', train=True, transform=transform, download=False)
    test_dataset = datasets.CIFAR10(root='../data', train=False, transform=transform)
    classes= tuple(train_dataset.classes)
    print(classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    X_train, y_train = Feature_Extractor(model,train_loader)
    X_test, y_test = Feature_Extractor(model,test_loader)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

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

if __name__ == "__main__":
    main()



