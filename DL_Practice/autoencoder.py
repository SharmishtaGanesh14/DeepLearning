import numpy as np
import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self,latent_dim,input_dim,hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,latent_dim),
            nn.ReLU()
        )
        self.Decoder=nn.Sequential(
            nn.Linear(latent_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,input_dim),
            nn.ReLU()
        )
    def forward(self,x):
        latent=self.encoder(x)
        reconstruction=self.Decoder(latent)
        return latent,reconstruction

random_seed=32
torch.manual_seed(random_seed)
np.random.seed(random_seed)
input_size=784
latent_size=32
hidden_size=128
input=torch.randn(1, input_size)
print(input)
autoenc=Autoencoder(latent_size,input_size,hidden_size)
print(autoenc(input)[1])