# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:06:35 2020

@author: BTLab
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# First load in the file
# Choosing hands behind head as its easily distinguishable

with open(r"C:\Users\BTLab\Documents\Aakash\bodies-at-rest=py3\bodies-at-rest-p3\data_BR\synth\hands_behind_head\test_roll0_plo_hbh_f_lay_set4_500.p", 'rb') as f:
    foo = pkl.load(f, encoding='latin1')

# Create dict to hold the angle information
angles = {'angle1': {}, 'angle2': {}, 'angle3': {}}

# Go through the angles and find parameters
for index, angle in enumerate(angles):
    angles[angle]['range'] = np.ptp(foo['joint_angles'][:][index])
    angles[angle]['mean'] = np.mean(foo['joint_angles'][:][index])
    angles[angle]['std'] = np.std(foo['joint_angles'][:][index])

# Hold angle 1 and 2 above mean + 1.96 * std and vary angle 3

# Neutral vecotr (default starting pos (belly button upwards))
# apply picth yaw roll to the vector 
# see if it matches with the corresponding image