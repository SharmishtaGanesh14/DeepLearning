import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pickle
from teacher_forcing_lstm import ImageCaptionLSTM

# ---------------- Load Vocabulary and Embeddings ----------------
with open("DataAndProcessing/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("DataAndProcessing/inv_vocab.pkl", "rb") as f:
    inv_vocab = pickle.load(f)

embedding_matrix = torch.tensor(np.load("DataAndProcessing/embedding_matrix.npy"), dtype=torch.float32)
vocab_size, embed_size = embedding_matrix.shape  # ← Automatically derived

# ---------------- Load Model ----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ImageCaptionLSTM(2048, embed_size, 512, vocab_size, embedding_matrix).to(device)
model.load_state_dict(torch.load("DataAndProcessing/image_caption_lstm.pth", map_location=device))
model.eval()
print("Model loaded successfully.")

# ---------------- Feature Extractor (ResNet50) ----------------
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------- Caption Generator ----------------
def generate_caption(model, img_feat, max_len=20, start_token_idx=1):
    model.eval()
    with torch.no_grad():
        outputs = model(img_feat.unsqueeze(0), max_len=max_len, teacher_forcing=False)
        predicted_ids = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()
        caption_tokens = []
        for idx in predicted_ids:
            word = inv_vocab.get(idx, "<UNK>")
            if word == "<END>":
                break
            caption_tokens.append(word)
        return " ".join(caption_tokens)

# ---------------- Inference on New Image ----------------
new_image_path = "img.png"
img = Image.open(new_image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    features = resnet(img_tensor).squeeze().cpu().numpy()

img_feat_tensor = torch.tensor(features, dtype=torch.float32).to(device)

caption = generate_caption(model, img_feat_tensor, max_len=20)
print("\nGenerated Caption:", caption)
