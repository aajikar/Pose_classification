# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:50:43 2021

@author: BTLab
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
from scipy.ndimage import median_filter
from skimage.transform import resize
from torchvision import transforms
import matplotlib.pyplot as plt
import re

# Code for PMAT Dataloader

# The PMAT data is structured per patient (13)
# Each patient has 17 different poses
# Each of the 17 poses has some number of pressure frames (not the same)
# Each patient has a folder for them
# Each pose is a text file
# The metadata for each patient is saved as .p file in the main folder
# First will have to collect the metadata for each subject
# The metadata for each subject contains the following items
# Gender Height Weight


# First funtion to collect all the .p files into a dataframe
def collect_files(dir_name, ext='.p'):
    """
    Get all the files of a particular extension.

    Parameters
    ----------
    dir_name : str
        Noramlized string for the directory name.
    ext : str, optional
        The extension of the files . The default is '.p'.

    Returns
    -------
    files : list
        List containig file paths of all the files.

    """
    # Go through the directory and get the names of all the files with
    # the specified extension
    files = []
    for file in os.listdir(dir_name):
        if file.endswith(ext):
            files.append(os.path.join(dir_name, file))

    return files


# Next a function to collect all the pressure arrays
def get_pres_arr(dir_name):
    """
    Get all pressure files of one subject in a directory.

    Returns a dictionary that contains keys for each pose. Each pose contains a
    numpy array that has the shape (frame, 64, 32).

    Parameters
    ----------
    dir_name : str
        Directory where the pressure files are located.

    Returns
    -------
    pres_dat : dict
        Dictionary of pose, each containing pressure data.

    """
    # First collect the names of all files in the directory
    file_names = collect_files(dir_name, ext='.txt')
    # Open each file reshape the array, and append to a list of dict
    # Create a list of dict
    pres_dat = {f'Pose {i+1}': [] for i in range(len(file_names))}
    pose = 1
    for file in file_names:
        arr = np.loadtxt(file)
        # Reshape the array
        # The original arrays are in the form [frame, (colxrow)]
        # Reshape the arrays so they are 64x32 in size
        arr = arr.reshape((-1, 64, 32))
        pres_dat[f'Pose {str(pose)}'] = arr
        pose += 1
    return pres_dat


# Next Function to get the pressure data for each subject
def get_each_sub_dat(dir_name):
    """
    Get pressure data for each subject.

    The pressure data for each subject is contained in a dictionary. The dict
    has the ID of the subjects. Each ID contains a dict which has the pose.

    Parameters
    ----------
    dir_name : str
        Path of the directory where the subject folders are located.

    Returns
    -------
    sub_dict : dict
        Dictionary containing subjects which contain poses which contain
        pressure data.

    """
    # First get all the directories in the root directory
    dirs = []
    for root, folders, _ in os.walk(dir_name):
        for name in sorted(folders):
            dirs.append(os.path.join(root, name))
    
    # Create a dictionary for each subject
    sub_dict = {f'S{i+1}' : {} for i in range(len(dirs))}
    
    # For each subject load in the pressure arrays
    for name in sub_dict.keys():
        # Get the pressure data
        for folder in dirs:
            if folder.endswith(name):
                sub_dict[name] = get_pres_arr(folder)
    
    return sub_dict


# Function to save each array and collect information about subject and pose
def save_data(sub_dict, root_dir):
    # Make dict for sub and pose
    metadata = {'Sub ID': [], 'Pose': [], 'Filename': []}
    # Go through each subject
    for subject in sub_dict.keys():
        # Go through each pose
        for pose in sub_dict[subject].keys():
            # Go through each array
            for array in range(len(sub_dict[subject][pose])):
                fn = root_dir / str(subject) / str(pose) / str(array)
                dir_name = root_dir / subject / pose
                metadata['Filename'].append(str(fn))
                metadata['Pose'].append(pose)
                metadata['Sub ID'].append(subject)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                with open(fn, 'wb') as f:
                    np.save(f, sub_dict[subject][pose][array])
    metadata = pd.DataFrame(metadata)
    return metadata


# Function to read, re-org, and save the data
def refactor_PMAT(original_dir, new_dir):
    # First get the data from the original directory
    sub_dict = get_each_sub_dat(original_dir)
    metadata = save_data(sub_dict, new_dir)
    return metadata


class PMAT_dataset(Dataset):
   
    def __init__(self, csv_file, transform=None, use_PMAT_labels=False):
        self.METADATA = pd.read_csv(csv_file)
        self.LABELS = self.METADATA['Pose']
        self.transform = transform
        self.PMAT_labels = use_PMAT_labels
        
    def __len__(self):
        return len(self.METADATA)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Load in the image and change to 3 channels
        img_name = self.METADATA.iloc[idx, 2]
        img = np.load(img_name)
        img = img.astype(np.float32)
        # Median filter the image
        img = median_filter(img, size=(3, 3))
        
        img = np.stack((img,)*3, axis=-1)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        
        # img = np.expand_dims(img, 0)
        
        # Normalize the image
        img = img / 1000 * 100
        
        # Change the pose to supine [1 0 0], lateral [0 1 0], prone [0 0 1]
        # Poses 1, 8, 9, 10, 11, 12, 15, 16, 17 are supine
        # Rest are lateral
        if int(self.LABELS[idx].split(' ')[1]) in [1, 2, 3, 4, 7, 8, 9, 16, 17]:
            label = 0
        else:
            label = 1

        one_hot_label = np.eye(3)[label]
        
        if self.PMAT_labels:
            one_hot_label = self.set_labels(idx)

        sample = {'image': torch.from_numpy(img),
                  'label': torch.from_numpy(one_hot_label)}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def print_sample_info(self, idx):
        print(f"Filename: {self.METADATA.iloc[idx, 2]}")
        print(f'Pose: {self.LABELS[idx]}')
        if int(self.LABELS[idx].split(' ')[1]) in [1, 2, 3, 4, 7, 8, 9, 16, 17]:
            label = 0
        else:
            label = 1
        print(f'Label: {label}')
        return None
    
    def plot_single_img(self, idx):
        img_name = self.METADATA.iloc[idx, 2]
        img = np.load(img_name)
        plt.imshow(img)
        return None
    
    def get_label(self, idx):
        if int(self.LABELS[idx].split(' ')[1]) in [1, 2, 3, 4, 7, 8, 9, 16, 17]:
            label = 0
        else:
            label = 1
        return label

    def set_labels(self, idx):
        label = int(self.LABELS[idx].split(' ')[1]) - 1
        one_hot_label = np.eye(17)[label]
        return one_hot_label

    def get_sub_ID(self, idx):
        fn = Path(self.METADATA.iloc[idx, 2])
        m = re.search(r'\d+$', str(fn.parent.parent))
        if m is not None:
            return int(m.group())
        else:
            raise ValueError('No subject ID could be found')
    
    def remove_sub(self, sub_id):
        # Go through the entire dataset and remove all cols with specified sub
        df_train = self.METADATA[int(re.search(r'\d+$', str(Path(self.METADATA['Filename']).parent.parent)).group()) != sub_id]
        df_val = self.METADATA[int(re.search(r'\d+$', str(Path(self.METADATA['Filename']).parent.parent)).group()) == sub_id]
        return df_train, df_val
    
    def get_LOSO_split(self, sub_id):
        train_indices = []
        val_indices = []
        
        for i in range(len(self.METADATA)):
            if self.get_sub_ID(i) != sub_id:
                train_indices.append(i)
            else:
                val_indices.append(i)
        
        return train_indices, val_indices

class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 0, 1)
        # Shape is now (64, 32, 3)
        image = resize(image, self.output_size)

        return {'image': image, 'label': label}


class ToTensor(object):
     def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label}


transform = transforms.Compose([Rescale((64, 27)), ToTensor()])

def bed_inclination_metadata(root, stem):
    # First make a list of all the files in the folder
    p = stem.glob('**/*')
    lst_of_files = [x for x in p if x.is_file()]
    
    # Make a dataframe
    df = pd.DataFrame()
    df['Filename'] = lst_of_files
    df['Subject'] = np.nan
    df['Pose'] = np.nan
        
    # Figure out which files are which subject
    for i in range(len(df['Filename'])):
        for j in range(1, 9):
            if f'S{j}' in df['Filename'][i].stem:
                df['Subject'][i] = j
        for k in range(1, 8):
            if f'F{k}' in df['Filename'][i].stem:
                df['Pose'][i] = k
    
    # Save the metadata in the root directory
    df.to_csv(root / 'metadata.csv', index=False)
    
    return None


class BedInclination(Dataset):
    
    def __init__(self, csv_file, sel_pose=1):
        self.METADATA = pd.read_csv(csv_file)
        self.POSE = sel_pose
        self.SUBJECT = self.METADATA['Subject']
        
        # Drop all poses other than the selected
        self.one_pose_df = self.METADATA[self.METADATA.Pose == self.POSE]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load in the image and change to 3 channels
        img_name = self.one_pose_df.iloc[idx, 0]
        img = np.genfromtxt(img_name)
        img = img.astype(np.float32)
        # Median filter the image
        img = median_filter(img, size=(3, 3))
        
        img = np.stack((img,)*3, axis=-1)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        
        # img = np.expand_dims(img, 0)
        
        # Normalize the image
        img = img / 500 * 100
        
        # All poses are supine
        one_hot_label = np.eye(3)[0]
        
        sample = {'image': torch.from_numpy(img),
                  'label': torch.from_numpy(one_hot_label)}

        return sample

    def __len__(self):
        return len(self.one_pose_df)


if __name__ == '__main__':
    # ds = PMAT_dataset(r"C:\Users\BTLab\Documents\Aakash\PMAT\metadata.csv",
    #                   transform=transforms.Compose([Rescale((64, 27)),
    #                                                 ToTensor()]))                                                        
    # idx = 20000
    # ds.print_sample_info(idx)
    # ds.plot_single_img(idx)
    # ID = ds.get_sub_ID(idx)
    # print('Subject ID:', ID)
    ds = BedInclination(r'C:\Users\BTLab\Documents\Aakash\PMAT Experiment ii\Bed Inclination\metadata.csv')
    a = ds.__getitem__(4)