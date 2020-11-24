# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:43:50 2020

@author: BTLab
"""

import pose
from torch.utils.data import DataLoader
import pandas as pd
import os


# Create a single metadata file that holds all the filenames and labels
def create_single_metadata_file(root_dir):
    # First make a list of all the CSV files and their pathnames
    my_dataset = pose.refactored_pose_dataset(root_dir)
    # Create a dict for poses
    # Contains the indices of lines which those pose are
    pose_dict = {'Lateral': [],
                 'Supine': [],
                 'Prone': []}
    mod_df = []
    # Load in one dataframe at a time
    for p in range(len(my_dataset.METADATA)):
        # Load single dataframe
        old_df = my_dataset.METADATA[p]
        print(len(old_df))
        # Load in single filename
        old_fn = my_dataset.METADATA_FILE_NAMES[p]
        size = len(old_df)
        # Figure out which pose the file belongs to
        if 'Supine' in old_fn:
            pose_dict['Supine'].append(p)
            label = ['supine'] * size
        elif 'Lateral' in old_fn:
            pose_dict['Lateral'].append(p)
            label = ['lateral'] * size
        else:
            pose_dict['Prone'].append(p)
            label = ['prone'] * size
        # each data frame append a column named pose type
        # First make a list size of samples in dict
        old_df['Pose Type'] = label
        # Now get the filename for each of the files
        img_filenames = []
        for i in range(len(old_df)):
            if i < 10:
                padded_idx = '0000' + str(i)
            elif i < 100:
                padded_idx = '000' + str(i)
            elif i < 1000:
                padded_idx = '00' + str(i)
            elif i < 10000:
                padded_idx = '0' + str(i)
            else:
                padded_idx = str(i)
            img_fn = os.path.join(os.path.dirname(old_fn),
                                  str('images'), padded_idx + '.npy')
            img_filenames.append(img_fn)
        # Add these to filenames
        old_df['Filename'] = img_filenames
        # Append to new dataframe
        mod_df.append(old_df)
    new_df = pd.concat(mod_df)
    return pose_dict, new_df


if __name__ == '__main__':
    root_dir = \
        r"C:\Users\BTLab\Documents\Aakash\Pose Classification\Data\Train"
    my_dict, my_df = create_single_metadata_file(root_dir)
    fn = os.path.join(root_dir, 'metadata.csv')
    my_df.to_csv(fn, index=False)
