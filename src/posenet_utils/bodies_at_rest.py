# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:11:10 2020

@author: BTLab
"""
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import proj_constants as pjc
from torchvision.transforms import RandomHorizontalFlip
from torchvision import transforms
import sklearn

class BodiesAtRestDataset(Dataset):
    """Dataset class for bodies at rest synthetic dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        self.METADATA = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        if csv_file.name == pjc.md8c:
            self.alpha = True
        else:
            self.alpha = False
        if self.alpha:
            self.LABELS = self.METADATA['Pose Type']
        else:
            self.LABELS = {'supine': 0, 'lateral': 1, 'prone': 2}

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.

        """
        return len(self.METADATA)

    def __getitem__(self, idx):
        """
        Get sample at specified index of dataset.

        Parameters
        ----------
        idx : tensor, list
            Index or indices of samples that are to be retrieved from dataset.

        Returns
        -------
        sample : TYPE
            DESCRIPTION.

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.METADATA.iloc[idx, 10]
        img = np.load(img_name, allow_pickle=True)
        img = img[0]
        img = np.reshape(img.astype(np.float32), (1, 64, 27))

        metadata = self.METADATA.iloc[idx, 0:9]
        metadata = np.array(metadata)

        if  not self.alpha:
            for label in self.LABELS:
                if label == self.METADATA.iloc[idx, 9]:
                    one_hot_label = np.eye(len(self.LABELS))[self.LABELS[label]]
                    one_hot_label = one_hot_label.astype(np.uint8)
        else:
            one_hot_label = np.eye(8)[self.LABELS[idx]]
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
    

class BodiesAtRest(torch.utils.data.IterableDataset):
    
    def __init__(self, csv_file, root_dir, transform=None, aug=10000):
        self.METADATA = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.LABELS = self.METADATA['Pose Type']
        self.aug_thresh = aug
        self.num_classes_dict = self.METADATA['Pose Type'].value_counts().to_dict()

    def __iter__(self):
        augs = transforms.Compose([RandomHorizontalFlip()])
        idx = -1
        while True:
            idx += 1
            samples = self.METADATA.groupby('Pose Type').sample(n=1)
            for _, sample in samples.iterrows():
                img_name = sample.iloc[10]
                img = np.load(img_name, allow_pickle=True)
                img = img[0]
                img = np.reshape(img.astype(np.float32), (1, 64, 27))
                
                metadata = sample.iloc[0:9]
                metadata = np.array(metadata)
                
                one_hot_label = np.eye(8)[sample.iloc[9]]
                one_hot_label = one_hot_label.astype(np.uint8)
                
                img = torch.from_numpy(img)
                
                if self.transform:
                    if self.num_classes_dict[sample.iloc[9]] < self.aug_thresh:
                        img = augs(img)
                        
                output = {'image': img,
                          'label': torch.from_numpy(one_hot_label)}
                
                yield output
                
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.METADATA.iloc[idx, 10]
        img = np.load(img_name, allow_pickle=True)
        img = img[0]
        img = np.reshape(img.astype(np.float32), (1, 64, 27))

        metadata = self.METADATA.iloc[idx, 0:9]
        metadata = np.array(metadata)
        
        one_hot_label = np.eye(8)[self.LABELS[idx]]
        one_hot_label = one_hot_label.astype(np.uint8)
        
        sample = {'image': torch.from_numpy(img),
                  'label': torch.from_numpy(one_hot_label)}
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def __len__(self):
        return len(self.METADATA)
    

class BodiesAtRestMultilabel(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.METADATA = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.LABELS = self.METADATA['Pose Type']
    
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
        
        # if label is supine_plo, hbh, xl, or sl it is also supine
        if self.LABELS[idx] in (1, 2, 3, 4):
            one_hot_label = np.eye(8)[self.LABELS[idx]] + np.eye(8)[0]
        # else if label is lateral_plo it is also lateral
        elif self.LABELS[idx] == 7:
            one_hot_label = np.eye(8)[self.LABELS[idx]] + np.eye(8)[6]
        # else the label is supine, phu, or lateral keep it as it is
        else:
            one_hot_label = np.eye(8)[self.LABELS[idx]]
        one_hot_label = one_hot_label.astype(np.uint8)

        sample = {'image': torch.from_numpy(img),
                  'label': torch.from_numpy(one_hot_label)}
        if self.transform:
            sample = self.transform(sample)

        return sample


class BodiesAtRestMultilableResNet(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None, aug_thresh=10000,
                 enable_binary=False, binary_class=None):
        self.METADATA = pd.read_csv(csv_file)
        self.METADATA = self.METADATA.sample(frac=1).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.LABELS = self.METADATA['Pose Type']
        self.num_classes_dict = self.METADATA['Pose Type'].value_counts().to_dict()
        self.aug_thresh = aug_thresh
        self.enable_binary = enable_binary
        self.binary_class = binary_class
    
    def __len__(self):
        return len(self.METADATA)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.METADATA.iloc[idx, 10]
        img = np.load(img_name, allow_pickle=True)
        img = img[0]
        img = np.reshape(img.astype(np.float32), (64, 27))
        img = img / 100
        img = np.stack((img,)*3, axis=-1)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        
        metadata = self.METADATA.iloc[idx, 0:9]
        metadata = np.array(metadata)
        
        # if label is supine_plo, hbh, xl, or sl it is also supine
        if self.LABELS[idx] in (1, 2, 3, 4):
            one_hot_label = np.eye(8)[self.LABELS[idx]] + np.eye(8)[0]
        # else if label is lateral_plo it is also lateral
        elif self.LABELS[idx] == 7:
            one_hot_label = np.eye(8)[self.LABELS[idx]] + np.eye(8)[6]
        # else the label is supine, phu, or lateral keep it as it is
        else:
            one_hot_label = np.eye(8)[self.LABELS[idx]]
        one_hot_label = one_hot_label.astype(np.uint8)
        
        augs = transforms.Compose([RandomHorizontalFlip()])
                        
        sample = {'image': torch.from_numpy(img),
                  'label': torch.from_numpy(one_hot_label)}
        
        if self.transform:
            img = torch.from_numpy(img)
            if self.num_classes_dict[np.argmax(one_hot_label)] < self.aug_thresh:
                img = augs(img)
            sample = {'image': img,
                      'label': torch.from_numpy(one_hot_label)}
            
        if self.enable_binary:
            if one_hot_label[self.binary_class]:
                one_hot_label = np.eye(2)[0]
            else:
                one_hot_label = np.eye(2)[1]

            sample = {'image': torch.from_numpy(img),
                      'label': torch.from_numpy(one_hot_label)}
            
            
        
        # if self.transform:
        #     sample = self.transform(sample)

        return sample


class BodiesAtRestResnet(Dataset):
    """Dataset class for bodies at rest synthetic dataset using Resnet"""

    def __init__(self, csv_file, root_dir=None, transform=None):
        self.METADATA = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        if csv_file.name == pjc.md8c:
            self.alpha = True
        else:
            self.alpha = False
        if self.alpha:
            self.LABELS = self.METADATA['Pose Type']
        else:
            self.LABELS = {'supine': 0, 'lateral': 1, 'prone': 2}

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.

        """
        return len(self.METADATA)

    def __getitem__(self, idx):
        """
        Get sample at specified index of dataset.

        Parameters
        ----------
        idx : tensor, list
            Index or indices of samples that are to be retrieved from dataset.

        Returns
        -------
        sample : TYPE
            DESCRIPTION.

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.METADATA.iloc[idx, 10]
        img = np.load(img_name, allow_pickle=True)
        img = img[0]
        img = img / 100
        img = np.reshape(img.astype(np.float32), (64, 27))
        img = np.stack((img,)*3, axis=-1)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)

        metadata = self.METADATA.iloc[idx, 0:9]
        metadata = np.array(metadata)

        if  not self.alpha:
            for label in self.LABELS:
                if label == self.METADATA.iloc[idx, 9]:
                    one_hot_label = np.eye(len(self.LABELS))[self.LABELS[label]]
                    one_hot_label = one_hot_label.astype(np.uint8)
        else:
            one_hot_label = np.eye(8)[self.LABELS[idx]]
            one_hot_label = one_hot_label.astype(np.uint8)

        sample = {'image': torch.from_numpy(img),
                  'label': torch.from_numpy(one_hot_label)}
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    csv_file = r"C:\Users\BTLab\Documents\Aakash\Data_train_val_test\metadata_all_classes.csv"
    dataset = BodiesAtRestMultilableResNet(csv_file, pjc.TRAIN_DIR)
    sample = dataset.__getitem__(0)
