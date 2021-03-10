# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:33:58 2021

@author: BTLab
"""
from pathlib import Path
from preprocessing import XSNDataReader
from posenet_utils import PoseNet
import torch
from preprocessing.XSN import XSNDataset as XSN
from preprocessing.XSN import transform
import proj_constants as pjc
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
import multiprocessing


# First load the model that will be used for classifying
# TODO: move this to project constants
model_fn = Path(r'C:\Users\BTLab\Documents\Aakash\Pose Classification\Final Pose Classification Model\0000000009.tar')

# Change the below file name depending on what file needs to have this done
XSN_CSV = Path(r"C:\Users\BTLab\Documents\Aakash\Patient Data from Stroke Ward\Patient 0 No Thresh\PS0008R1S0023_20020101_070042_PSMLAB.csv")

metadata_dir = Path(r'C:\Users\BTLab\Documents\Aakash\Leo Pressure Data\Center Line')
new_data_dir = Path(r'C:\Users\BTLab\Documents\Aakash\Patient Data from Stroke Ward\Patient1\Data')
new_plot_dir = Path(r'C:\Users\BTLab\Documents\Aakash\Leo Pressure Data\Center Line\Plots')



# Next convert the original CSV file into a dataset structure
def create_dataset(original_csv, metadata_dir, new_data_dir):
    df = XSNDataReader.read_XSN_csv(original_csv)
    new_data, pres = XSNDataReader.convert_df(df)
    df_with_fn, metadata_fn = XSNDataReader.save_pres_data(metadata_dir, new_data_dir,
                                              pres, new_data)
    return metadata_fn


# Classify the dataset into supine positions
def classify_pose(model_fn, metadata_fn):
    # Load the torch model and the dict keys
    model = PoseNet.create_poseresnet_model()
    state_dict = torch.load(model_fn)['model']
    model.load_state_dict(state_dict)
    
    # Load the dataset
    dataset = XSN(metadata_fn, transform)
    batch_size = 128
    num_classes = 3
    loss_function = torch.nn.BCEWithLogitsLoss()
    
    # No need to shuffle data when testing
    test_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=16)
    
    # Create all necessary tensors for storing predictions and indices
    device = \
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    y_true = torch.tensor([])
    all_preds = torch.tensor([])
    all_preds.to(device)
    indices = torch.tensor([])
    indices.to(device)
    
    # Set the model in evaluation mode
    with torch.no_grad():
        for sample in tqdm(test_loader):
            
            # Load the device and make predictions
            imgs = sample['image'].to(device)
            y_pred = model(imgs)
            all_preds = torch.cat((all_preds.cpu(), y_pred.cpu()), dim=0)
            
            # Get the indices
            indices = torch.cat((indices, sample['index']), dim=0)
            
    return all_preds.cpu(), indices.cpu()


# Keep only the indices where the predictions are supine
# Label 0 is supine
def get_supine_indices(all_preds, indices):
    yp = torch.argmax(all_preds, dim=1)
    pose_indices = yp == 0
    sup_indices = indices[pose_indices]
    lat_indices = indices[~pose_indices]
    return sup_indices, lat_indices


# Add the label annotation to each frame in the dataset
# TODO: Do this after human verification
def label_dataframe(metadata_fn, labels, sup_indices, new_metadata_dir):
    # The dataset was not shuffled so the indices are in sorted order
    # load the data frame
    df = pd.read_csv(metadata_fn)
    # Create a new column for labels
    df['Labels'] = torch.argmax(labels, dim=1)
    # Save the data frame
    df.to_csv(metadata_fn, index=False)
    # Now drop frames that are not supine
    sup_df = df.drop(sup_indices)
    # Save the new data frame
    new_fn = new_metadata_dir / 'metadata_with_annotations.csv'
    sup_df.to_csv(new_fn)
    return sup_df, new_fn


# Calculate the COP for a given pressure array
def calculate_COP(pres_arr):
    x = np.arange(0, pres_arr.shape[0])
    COP = np.empty(pres_arr.shape[1])
    for i in range(pres_arr.shape[1]):
        COP[i] = np.sum(x * pres_arr[:, i]) / np.sum(pres_arr[:, i])
    return COP


# Plot the given array with COP line through it
def plot_and_save_annotated_img(filename, COP, new_data_dir, deg):
    # Load the pressure array
    pres_arr = np.load(filename)
    filename = Path(filename)

    # Second create a model only if COP is not nan for all values
    mask = ~np.isnan(COP)
    if not True in mask:
        pass
    else:
        myline = np.linspace(0, pres_arr.shape[1]-1, pres_arr.shape[1])
        mymodel = np.poly1d(np.polyfit(np.arange(0, pres_arr.shape[1])[mask], COP[mask], deg))
        
        # Next plot the image
        plt.imshow(pres_arr)
        plt.plot(myline, mymodel(myline))
        plt.axis('off')
        fn = new_data_dir / str(filename.stem + '.png')
        plt.savefig(fn)
    
    return None


def classify_and_annotate_data(XSN_CSV, metadata_dir, new_data_dir, model_fn):
    # First create the dataset from incoming data
    # metadata_fn = create_dataset(XSN_CSV, metadata_dir, new_data_dir)
    
    metadata_fn = Path(r'C:\Users\BTLab\Documents\Aakash\Patient Data from Stroke Ward\Patient 0 No Thresh\metadata.csv')
    
    # Next perform pose classification
    all_preds, indices = classify_pose(model_fn, metadata_fn)
    
    # Get only the supine indices
    sup_indices, lat_indices = get_supine_indices(all_preds, indices)
    
    # Save the labelled metadata files
    # sup_df, sup_df_fn = label_dataframe(metadata_fn, all_preds, lat_indices,
    #                                     new_metadata_dir=new_data_dir.parent)
    
    # # Calculate COP for all the frames
    # all_COP = []
    # for file in sup_df['Filename']:
    #     fn = file + '.npy'
    #     pres_arr = np.load(fn)
    #     COP = calculate_COP(pres_arr)
    #     all_COP.append(COP)
    
    # # Plot and save the images
    # # Parallel(n_jobs=-1)(delayed(plot_and_save_annotated_img)(file, COPx, new_data_dir, deg=2) for file sup_df['Filename'] for )
    # # p1 = multiprocessing.Process(target=plot_and_save_annotated_img, args=(10, ))
    # # num_cores = multiprocessing.cpu_count()
    # # Parallel(n_jobs=num_cores)(delayed(plot_and_save_annotated_img)(sup_df['Filename'][i], all_COP[i], new_plot_dir, 2) for i in range(len(sup_df['Filename'])))
    # for i in range(len(sup_df['Filename'])):
    #     plot_and_save_annotated_img(sup_df['Filename'][i], all_COP[i], new_plot_dir, 2)
    
    # # Save the COP to the metadata file
    # sup_df['COP'] = all_COP
    # sup_df.to_csv(sup_df_fn)
    
    return sup_indices, lat_indices


def plot():
    dir_name = Path(r'C:\Users\BTLab\Documents\Aakash\Leo Pressure Data\Center Line\Plots')
    df = pd.read_csv(r'C:\Users\BTLab\Documents\Aakash\Leo Pressure Data\Center Line\metadata_with_annotations.csv')
    for i in range(len(df)):
        fn = df['Filename'][i] + '.npy'
        fn = Path(fn)
        img = np.load(fn)
        COP = calculate_COP(img)
        mask = ~np.isnan(COP)
        myline = np.linspace(0, 117, 118)
        mymodel = np.poly1d(np.polyfit(np.arange(0, 118)[mask], COP[mask], 2))
        plt.imshow(img)
        plt.plot(myline, mymodel(myline))
        plt.axis('off')
        # new_fn = dir_name / str(fn.stem + '.png')
        # plt.savefig(new_fn)


if __name__ == '__main__':
    start_time = time.time()
    sup, lat = classify_and_annotate_data(XSN_CSV, metadata_dir, new_plot_dir, model_fn)
    print("--- %s seconds ---" % (time.time() - start_time))
    # dir_name = Path(r'C:\Users\BTLab\Documents\Aakash\Leo Pressure Data\Center Line\Plots')
    # df = pd.read_csv(r'C:\Users\BTLab\Documents\Aakash\Leo Pressure Data\Center Line\metadata_with_annotations.csv')
    # for i in range(len(df)):
    #     fn = df['Filename'][i] + '.npy'
    #     fn = Path(fn)
    #     img = np.load(fn)
    #     COP = calculate_COP(img)
    #     mask = ~np.isnan(COP)
    #     myline = np.linspace(0, 117, 118)
    #     mymodel = np.poly1d(np.polyfit(np.arange(0, 118)[mask], COP[mask], 2))
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.plot(myline, mymodel(myline))
    #     plt.axis('off')
    #     new_fn = dir_name / str(fn.stem + '.png')
    #     plt.savefig(new_fn)
