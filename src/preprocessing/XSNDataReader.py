# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:20:41 2021

@author: BTLab
"""

import pandas as pd
from datetime import datetime
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os


# Function to read the csv as pandas table and change the header to one name
def read_XSN_csv(fn):
    df = pd.read_table(fn)
    df.rename(columns={df.columns[0]: 'dat'}, inplace=True)
    return df




# Create a pandas data frame with the following fields
# Frame    Date    Time    COP Row     COP Col    Avg Pressure    Peak Pressure    Contact Area

# The first 16 elements index 0-15 are metadata
# Collect the required metadat from their and put them in a data frame
# All data is saved as strings, so change to the necessary type

def convert_df(df):
    # First make a dict of the things needed
    new_dat = {'Frame': [], 'Datetime': [], 'COP': [],
               'Avg Pres': [], 'Peak Pres': [], 'Contact Area': []}
    # Next an empty array
    pres = []
    # The first frame is at index 2
    # The next frame is at index 64, 126../
    counter = 0
    for i in range(len(df)):
        if i % 62 == 2:
            new_dat['Frame'].append(int(df['dat'][i].split(',')[1]))
        elif i % 62 == 3:
            new_dat['Datetime'].append(datetime.strptime(df['dat'][i].split(',')[1] + df['dat'][i+1].split(',')[1], '"%Y %b %d""%H:%M:%S"'))
        elif i % 62 == 8:
            new_dat['COP'].append(tuple([float(df['dat'][i].split(',')[1]), float(df['dat'][i+1].split(',')[1])]))
        elif i % 62 == 12:
            new_dat['Avg Pres'].append(float(df['dat'][i].split(',')[1]))
        elif i % 62 == 13:
            new_dat['Peak Pres'].append(float(df['dat'][i].split(',')[1]))
        elif i % 62 == 14:
            new_dat['Contact Area'].append(float(df['dat'][i].split(',')[1]))
        
        # From index 16-63 is pressure data
        row = None
        if i % 62 == 16:
            for j in range(48):
                if row is None:
                    row = [float(k) for k in df['dat'][i+j].split(',')]
                else:
                    row = np.array(row)
                    row = np.vstack([row, np.array([float(k) for k in df['dat'][i+j].split(',')])])
            # At the end of the for loop increase i by j
            counter += 47
        
        # Append row to pres
        if row is not None:
            pres.append(row)
        
        # Increase counter by 1 at the end of the loop
        counter += 1
    
    # Convert new_dat to a dataframe
    new_dat = pd.DataFrame.from_dict(new_dat)
    
    return new_dat, pres





# Function to save each file in the specified folder
# zfill each file number with 6 zeros
def save_pres_data(metadata_dir, pres_dir, pres_dat, df):
    lst_of_fn = {'Filename': []}
    # Save each array in the list
    for arr in range(len(pres_dat)):
        fn = pres_dir / (str(arr).zfill(6) + '.npy')
        np.save(fn, pres_dat[arr])
        lst_of_fn['Filename'].append(fn)
    # add the filename col to dataframe
    df['Filename'] = lst_of_fn['Filename']
    metadata_fn = metadata_dir / ('metadata.csv')
    df.to_csv(metadata_fn, index=False)
    return df, metadata_fn



def generate_video(og_dir, new_dir):
    # First make a list of all the file names in the original directory
    p = og_dir.glob('**/*')
    lst_of_files = [x for x in p if x.is_file()]
    
    # Next plot each image sequentially in new dir and save as png
    for fn in lst_of_files:
        plot_and_save(fn, new_dir, add_threshold=True)
    
    return None

def plot_and_save(fn, new_dir, add_threshold=False):
    img = np.load(fn)
    if add_threshold:
        threshold = 0.07
        super_threshold_indices = img < threshold
        img[super_threshold_indices] = 0
    plt.imshow(img)
    plt.axis('off')
    new_fn = new_dir / str(fn.stem + '.png')
    plt.savefig(new_fn)
    return None


if __name__ == '__main__':
    # First get the filenames
    # Start parsing the data
    # First load in the data
    fn = Path(r"C:\Users\BTLab\Documents\Aakash\Patient Data from Stroke Ward\p6\PS0008R4S0039_20210208_221341_PSMLAB.csv")
    
    # Next read the file as a data frame
    df = read_XSN_csv(fn)
        
    new_dat, pres = convert_df(df)
    
    meta_dir = Path(r'C:\Users\BTLab\Documents\Aakash\Patient Data from Stroke Ward\p6')
    pres_dir = Path(r'C:\Users\BTLab\Documents\Aakash\Patient Data from Stroke Ward\p6\Data')
    
    df_with_fn = save_pres_data(meta_dir, pres_dir, pres, new_dat)