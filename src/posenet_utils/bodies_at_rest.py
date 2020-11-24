# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:11:10 2020

@author: BTLab
"""
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

class BodiesAtRestDataset(Dataset):
    
    LABELS = {'supine': 0, 'lateral': 1, 'prone': 2}
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.METADATA = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.METADATA)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.METADATA.iloc[idx, 10]
        img = np.load(img_name, allow_pickle=True)
        img = img[0]
        img = np.reshape(img.astype(np.float32), (1, 64, 27))

        metadata = self.METADATA.iloc[idx, 0:9]
        metadata = np.array(metadata)

        for label in self.LABELS:
            if label == self.METADATA.iloc[idx, 9]:
                one_hot_label = np.eye(len(self.LABELS))[self.LABELS[label]]
                one_hot_label = one_hot_label.astype(np.uint8)

        sample = {'image': torch.from_numpy(img),
                  'label': torch.from_numpy(one_hot_label)}
        if self.transform:
            sample = self.transform(sample)

        return sample


def view_one_img(idx, dataset):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    fig = plt.figure()
    sample = dataset[idx]
    print(idx, sample['image'].shape, sample['label'].shape)
    ax = plt.subplot(1, 1, 1)
    plt.tight_layout()
    plt.imshow(np.reshape(sample['image'], (64,27)))
    ax.set_title('Sample #{}'.format(idx))
    ax.axis('off')

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = (sample['image'],
                                  sample['label'])

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
    

