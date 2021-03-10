# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 08:37:36 2020

@author: Aakash
"""
import os
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_pickle(filename):
    """
    Load pickle file using latin1 encoding.

    Parameters
    ----------
    filename : str
        Normalized location of file that needs to be opened.

    Returns
    -------
    bytes
        The values within the loaded file.

    """
    with open(filename, 'rb') as f:
        return pkl.load(f, encoding='latin1')


def get_file_names(dirname):
    """
    Get name of all the files in the specified directory.

    Parameters
    ----------
    dirname : str
        Root directory name.

    Returns
    -------
    list_of_files : list
        List of file names as normalized strings.

    """
    list_of_files = []
    for root, dirs, files in os.walk(dirname):
        for name in files:
            list_of_files.append(os.path.join(root, name))
    return list_of_files


def get_pose_type(filename):
    """
    Get the pose that is in the filename.

    Parameters
    ----------
    filename : str
        Name of file that needs pose to be determined.

    Returns
    -------
    pose : str
        Name of pose; supine, or lateral.

    """
    poses = {}
    poses['supine'] = ['crossed', 'straight', 'supine', 'behind']
    poses['lateral'] = 'rollpi'
    poses['prone'] = 'prone'
    for pose in poses:
        for i in range(len(poses[pose])):
            if poses[pose][i] in filename:
                return pose


def get_key(val, my_dict):
    """
    Get the key for which a value is in.

    Parameters
    ----------
    val : int, float, str, bool
        Accepts multiple types of data to be found in the dictionary.
    my_dict : dict
        Dictionary for which index needs to be found for value.

    Returns
    -------
    key : str
        Name of the key that holds the value. If value does not exist, then
        "key doesn't exist" is returned.

    """
    for key, value in my_dict.items():
        if val in value:
            return key

    return "key doesn't exist"


class pose_dataset():
    """Class to hold the pose data."""

    def __init__(self):
        self.train_data = []
        self.labels = []
        self.list_of_files = []

    def get_labels(self, dirname):
        """
        Append labels to the object for all the file names.

        Parameters
        ----------
        dirname : str
            Directory where the data files are located.

        Returns
        -------
        None.

        """
        self.list_of_files = get_file_names(dirname)
        pose_to_label_dict = {}
        pose_to_label_dict['no_patient'] = 0
        pose_to_label_dict['supine'] = 1
        pose_to_label_dict['lateral'] = 2
        pose_to_label_dict['prone'] = 3
        for file in self.list_of_files:
            # For each file get a corresponding label
            # The label is the same for all the samples in that file
            # Indices match between files and labels
            self.labels.append(pose_to_label_dict[get_pose_type(file)])

    def separate_metadata(self, idx):
        """
        Seperate metadata from original data before refactoring.

        Parameters
        ----------
        idx : int
            Index of metadata that needs to be refactored.

        Returns
        -------
        None.

        """
        # idx, choose a single file to load
        # The loaded file will be a dictionary
        # Convert it to DataFrame right away
        self.metadata = \
            pd.DataFrame.from_dict(load_pickle(self.list_of_files[idx]))
        # Separate the dataframes into metadata and pressure data
        # Train data is just ['images']
        self.train_data = \
            self.metadata.drop(self.metadata.columns.difference(['images']), 1)
        # The rest is metadata
        self.metadata.drop(columns='images', inplace=True)

    # Function to save all samples one at a time in pressure array
    def save_pres_dat(self, dirname):
        """
        Save the numpy arrays with ascending numerical names.

        The files are padded with leading zeros. Since the maximum number of
        samples is 16000, padding is done for 9999.

        Parameters
        ----------
        dirname : str
            Location of directory where the numpy arrays should be stored.

        Returns
        -------
        None.

        """
        arrs = self.train_data.to_numpy()
        ext = '.npy'
        for i in range(arrs.shape[0]):
            if i < 10:
                sample_num = "0000" + str(i)
            elif i < 100:
                sample_num = "000" + str(i)
            elif i < 1000:
                sample_num = "00" + str(i)
            elif i < 10000:
                sample_num = "0" + str(i)
            else:
                sample_num = str(i)
            fn = os.path.join(dirname, (sample_num+ext))
            np.save(fn, arrs[i])
        print(f"{arrs.shape[0]} files saved to {dirname}")

    # Function to save metadata as CSV file
    def save_metadata(self, dirname):
        """
        Save the metadata as a CSV file in the specified directory.

        Parameters
        ----------
        dirname : str
            Location of directory where CSV file should be saved.

        Returns
        -------
        None.

        """
        file = 'metadata.csv'
        fn = os.path.join(dirname, file)
        self.metadata.to_csv(fn, index=False)

    # Function to figure out if the data is part of test or train
    def test_or_train(self):
        """
        Determine if data belongs to training or testing set.

        Returns
        -------
        None.

        """
        self.train_test_idx = {}
        self.train_test_idx['train'] = []
        self.train_test_idx['val'] = []
        for i in range(len(self.list_of_files)):
            if 'train' in self.list_of_files[i]:
                self.train_test_idx['train'].append(i)
            else:
                self.train_test_idx['val'].append(i)

    # Function to create directory
    def create_dir(self, dirname):
        """
        Make a new directory if one does not already exist.

        Parameters
        ----------
        dirname : str
            Location of directory that needs to be created.

        Returns
        -------
        None.

        """
        # Check if dir exists or not
        if not os.path.exists(dirname):
            os.mkdir(dirname)

    # Function to open one file
    def load_data(self, filename):
        dat = load_pickle(filename)
        # Separate pressure and metadata

    # Funtion to open file at a time, then separate the metadata and then
    # save the metadata and numpy arrays in the sepcified directory
    def refactor_data(self, old_dir, new_dir):
        """
        Change the way data is structured.

        The data is changed such that pressure samples are seperated into
        individual files. The data is seperated into train and test directories
        and in each directory folder for supine, lateral, and prone poses, and
        within those are the subpose type. Each subpose has metadata associated
        with it. Within each subpose folder is a directory called images which
        contains the numpy arrays for the pressure images.

        Parameters
        ----------
        old_dir : str
            Location of the directory where the old data is located.
        new_dir : str
            Location of the directory where the new refactored data needs to be
            stored.

        Returns
        -------
        None.

        """
        # First acquire all filename in the old directory and give them labels
        self.get_labels(old_dir)
        # Next get all the train test labels
        self.test_or_train()
        # For each file in the list
        for file_idx in range(len(self.list_of_files)):
            # Open the single file and seperate the metadata
            self.separate_metadata(file_idx)
            # Make new dir for train or test
            # Find if the current file is train or test
            train_test = get_key(file_idx, self.train_test_idx)
            sub_dir1 = os.path.join(new_dir, train_test)
            self.create_dir(sub_dir1)
            sub_dir2 = os.path.join(sub_dir1, str(self.labels[file_idx]))
            self.create_dir(sub_dir2)
            # Create dir for pose
            pose_name = os.path.basename(self.list_of_files[file_idx])
            pose_dir = os.path.join(sub_dir2, pose_name)
            self.create_dir(pose_dir)
            self.save_metadata(pose_dir)
            pres_img_dir = os.path.join(pose_dir, "images")
            self.create_dir(pres_img_dir)
            self.save_pres_dat(pres_img_dir)


# Code after this deals with data that has already been refactored

class refactored_pose_dataset(Dataset):
    """Dataset class for the refactored data."""

    def __init__(self, root):
        self.ROOT_DIR = root
        self.get_metadata_files()
        self.get_metadata()

    def get_metadata(self):
        """
        Load metadata from each file into a list.

        Returns
        -------
        None.

        """
        self.METADATA = []
        for file in self.METADATA_FILE_NAMES:
            self.METADATA.append(pd.read_csv(file))

    # Get all the filenames of metadata CSVs
    def get_metadata_files(self):
        """
        Make a list of all CSV metadata filenames in the dataset.

        Returns
        -------
        None.

        """
        self.METADATA_FILE_NAMES = []
        for root, dirs, files in os.walk(self.ROOT_DIR):
            for file in files:
                if file.endswith('.csv'):
                    self.METADATA_FILE_NAMES.append(os.path.join(root, file))

    def __getitem__(self, sample_idx, file_idx):
        """
        Get single sample with data and metadata.

        Parameters
        ----------
        sample_idx : tensor
            Index of the current sample that needs to be loaded.
        file_idx : tensor
            Index of the current file that the sample needs to be loaded from.

        Returns
        -------
        sample : dict
            images : contains the pressure data.
            metadata : contains the metadata.

        """
        if torch.is_tensor(sample_idx):
            sample_idx = sample_idx.tolist()
        if torch.is_tensor(file_idx):
            file_idx = file_idx.tolist()
        img_name = self.get_image_name(sample_idx, file_idx)
        image = np.load(img_name)
        metadata = self.METADATA[file_idx].iloc[sample_idx]
        metadata = np.array([metadata])
        sample = {'image': image, 'metadata': metadata}
        return sample

    def __len__(self, file_idx):
        """
        Return the number of samples in the current metadata file.

        Parameters
        ----------
        file_idx : tensor
            Index of the file that needs to be accessed.

        Returns
        -------
        int
            Number of items in the container.

        """
        if torch.is_tensor(file_idx):
            file_idx = file_idx.tolist()
        return len(self.METADATA[file_idx])

    # Function to make labels equal the number of samples
    def copy_labels(self):
        return None

    def get_image_name(self, sample_idx, file_idx):
        """
        Change sample index to a zero padded index.

        Parameters
        ----------
        sample_idx : list
            Index of the current sample that needs to be loaded.
        file_idx : list
             Index of the current file that the sample needs to be loaded from.

        Returns
        -------
        img_name : str
            Normalized string location of the image.

        """
        if sample_idx < 10:
            padded_sample_idx = "0000" + str(sample_idx)
        elif sample_idx < 100:
            padded_sample_idx = '000' + str(sample_idx)
        elif sample_idx < 1000:
            padded_sample_idx = '00' + str(sample_idx)
        elif sample_idx < 10000:
            padded_sample_idx = '0' + str(sample_idx)
        else:
            padded_sample_idx = str(sample_idx)
        img_name = \
            os.path.join(os.path.dirname(self.METADATA_FILE_NAMES[file_idx]),
                         padded_sample_idx)
        return img_name

    # Visualize specific image in dataset
    def viz_one_sample(self, sample_idx, file_idx):
        """
        Show specified pressure image and its metadata.

        Parameters
        ----------
        sample_idx : tensor
            Index of the current sample that needs to be loaded.
        file_idx : tensor
            Index of the current file that the sample needs to be loaded from.

        Returns
        -------
        None.

        """
        if torch.is_tensor(sample_idx):
            sample_idx = sample_idx.tolist()
        if torch.is_tensor(file_idx):
            file_idx = file_idx.tolist()
        img_name = self.get_image_name(sample_idx, file_idx)
        image = np.load(img_name)
        plt.imshow(image)
        print(self.METADATA[file_idx].iloc[sample_idx])


class ToTensor(object):
    """Convert ndarrays in sample to tensors."""

    def __call__(self, sample):
        """
        Convert numpy array in sample to tensor.

        Parameters
        ----------
        sample : dict
            images : contains the pressure data.
            metadata : contains the metadata.

        Returns
        -------
        dict
            Input dict now as tensors.

        """
        image, metadata = sample['image'], sample['metadata']
        return {'image': torch.from_numpy(image),
                'metadata': torch.from_numpy(metadata)}






if __name__ == '__main__':
    db = pose_dataset()
    db.refactor_data(r"C:\Users\BTLab\Documents\Aakash\bodeis-at-rest-data",
                      r"C:\Users\BTLab\Documents\Aakash\Data_train_val_test")




























































