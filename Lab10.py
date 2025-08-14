# Date created: 13/08/2024
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim:
# Successfully implement a deep learning model to solve a biological problem.
# Process biological datasets to develop predictive models

# Task: Downloaded the 1000 Genomes dataset (used in paper as val set to do the training using FFN)

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ThousandGenomesDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.thousandgenomes = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.thousandgenomes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.thousandgenomes.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.thousandgenomes.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


# batch norm
# pytorch should be included