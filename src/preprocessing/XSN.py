# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:30:44 2021

@author: BTLab
"""

from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from torchvision import transforms

# Dataloader object for XSENSOR data

class XSNDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.METADATA = pd.read_csv(csv_file)
        self.transform = transform
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load in the image and change to 3 channels
        img_name = self.METADATA.iloc[idx, 6]
        img_name = img_name + '.npy'
        img = np.load(img_name)
        img = img.astype(np.float32)
        # threshold = 0.07
        # super_threshold_indices = img < threshold
        # img[super_threshold_indices] = 0.0
        
        img = np.stack((img,)*3, axis=-1)
        img = np.swapaxes(img, 0, 2)
        
        # Normalize the image
        img = img / 2.66645 #* 100
        
        one_hot_label = np.eye(3)[0]
        
        sample = {'image': torch.from_numpy(img),
                  'label': torch.from_numpy(one_hot_label),
                  'index': torch.tensor(idx)}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
    def __len__(self):
        return len(self.METADATA)

    def plot_single_sample(self, idx):
        img_name = self.METADATA.iloc[idx, 6]
        # img_name = img_name + '.npy'
        img = np.load(img_name)
        img = img.astype(np.float32)
        plt.imshow(img)
        return None
    
    def predict_single_sample(self, model, idx, device):
        # Get single sample
        sample = self.__getitem__(idx)
        img = sample['image'].to(device)
        model.eval()
        y_pred = model(img)
        one_hot_label = torch.argmax(y_pred, 1)
        return y_pred, one_hot_label


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, index = sample['image'], sample['label'], sample['index']
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 0, 1)
        # Shape is now (118, 48, 3)
        image = resize(image, self.output_size)

        return {'image': image, 'label': label, 'index': index}


class ToTensor(object):
     def __call__(self, sample):
        image, label, index = sample['image'], sample['label'], sample['index']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label,
                'index': index}

transform = transforms.Compose([Rescale((64, 27)), ToTensor()])

if __name__ == '__main__':
    dataset = XSNDataset(r'C:\Users\BTLab\Documents\Aakash\Patient Data from Stroke Ward\Patient1\metadata.csv')
    a = dataset.__getitem__(0)
    dataset.plot_single_sample(12000)