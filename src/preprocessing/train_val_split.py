# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:37:41 2020

@author: BTLab
"""

# Manually make validation set
import os
import pickle as pkl


root = os.path.abspath(r'C:\Users\BTLab\Documents\Aakash\bodies-at-rest=py3\bodies-at-rest-p3\data_BR\synth\straight_limbs')
files = os.listdir(root)
train_files = []
for file in files:
    if 'train' in file:
        train_files.append(file)


for file in train_files:
    with open(os.path.join(root, file), 'rb') as f:
        dat = pkl.load(f, encoding='latin1')
    
    train_dat = {}
    val_dat = {}
    for key in dat.keys():
        train_dat[key] = dat[key][:int((len(dat[key])+1)*.8)]
        val_dat[key] = dat[key][int((len(dat[key])+1)*.8):]

    save_dir = r"C:\Users\BTLab\Documents\Aakash\bodeis-at-rest-data\straight_limbs"
    
    val_fn = file.replace("train", "val")
    
    with open(os.path.join(save_dir, file), 'wb') as f:
        pkl.dump(train_dat, f)
    
    with open(os.path.join(save_dir, val_fn), 'wb') as f:
        pkl.dump(val_dat, f)
