import os
import torch

import numpy as np

from torch.utils.data import Dataset, Subset
from PIL import Image

from types import MethodType
 
class BaseDataset(Dataset):

    def __init__(self, root_dir, labels=None, transform=None, target_transform=None, label_encoding=False, custom_encoding=None):
        super().__init__()
        self.root_dir = root_dir
        if labels:
            self.labels = labels
        else:
            self.labels = os.listdir(self.root_dir)

        self.transform = transform
        self.target_transform = target_transform

        # Creating dictionaries for label encoding and decoding
        self.label_encoding = label_encoding
        self.label_encoder = custom_encoding
        if self.label_encoding and self.label_encoder is None:
            self.label_encoder = dict()
            for i, label in enumerate(self.labels):
                self.label_encoder[label] = i
        if label_encoding:
            self.label_decoder = {v: k for k, v in self.label_encoder.items()}


        self.targets = []
        self.data_paths = []
        #
        if all(isinstance(label, int) for label in self.labels):
            self.labels = [int(label) for label in self.labels]

        for label in self.labels:
            for file in os.listdir(os.path.join(self.root_dir, str(label))):
                file_path = os.path.join(self.root_dir, str(label), file)
                self.data_paths.append(file_path)
                self.targets.append(label)




    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, idx):
        sample = Image.open(self.data_paths[idx])
        target = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        
        if self.label_encoding:
            target = self.label_encoder[target]

        if self.target_transform:
            target = self.target_transform(target)

        return sample, target
