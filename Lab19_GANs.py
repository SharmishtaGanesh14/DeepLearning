import torch
from torch import nn
import math
import matplotlib.pyplot as plt

torch.manual_seed(111)

# --- Create training data (sine wave points) ---
train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

plt.figure(figsize=(6, 4))
plt.title("Real training data (sine curve)")
plt.plot(train_data[:, 0], train_data[:, 1], ".")
plt.show()

# --- Dataloader ---
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.model(x)

# --- Generator ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, x):
        return self.model(x)

discriminator = Discriminator()
generator = Generator()

# --- Training setup ---
lr = 0.001
num_epochs = 1000
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

# --- Training Loop ---
for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Create fake data
        latent_space_samples = torch.randn((batch_size, 2))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))
        real_samples_labels = torch.ones((batch_size, 1))

        # Combine real + fake
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # --- Train Discriminator ---
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Compute discriminator accuracy
        predicted = (output_discriminator >= 0.5).float()
        D_acc = (predicted == all_samples_labels).float().mean().item()

        # --- Train Generator ---
        generator.zero_grad()
        latent_space_samples = torch.randn((batch_size, 2))
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # Compute generator accuracy (how well it fools D)
        G_acc = (output_discriminator_generated >= 0.5).float().mean().item()

    # --- Visualization every 50 epochs ---
    if epoch % 50 == 0 or epoch == num_epochs - 1:
        # with torch.no_grad():
        #     latent_space_samples = torch.randn(200, 2)
        #     generated_samples = generator(latent_space_samples)
        # plt.figure(figsize=(6, 4))
        # plt.title(f"Epoch {epoch}")
        # plt.plot(train_data[:, 0], train_data[:, 1], ".", label="Real Data", alpha=0.3)
        # plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".", label="Generated Data")
        # plt.legend()
        # plt.show()

        print(f"Epoch: {epoch:03d} | Loss D: {loss_discriminator.item():.4f} | Loss G: {loss_generator.item():.4f} | D_acc: {D_acc*100:.2f}% | G_acc: {G_acc*100:.2f}%")
