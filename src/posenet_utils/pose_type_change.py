# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:14:10 2020

@author: BTLab
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import proj_constants as pjc
from pathlib import Path

# with open (r"C:\Users\BTLab\Documents\Aakash\bodies-at-rest=py3\bodies-at-rest-p3\data_BR\real\S107\prescribed.p", 'rb') as f:
#     foo = pkl.load(f, encoding='latin1')

# bar = foo['RGB'][11]
# bar = bar[:, :, ::-1]
# plt.imshow(bar)
# plt.axis('off')
# plt.tight_layout()
# plt.savefig(r'C:\Users\BTLab\Documents\Aakash\Pose Classification\Plots\lateral.pdf', format='pdf')


def convert_to_all_poses():
    df = pd.read_csv(pjc.TRAIN_METADATA_CSV)
    pose_dict = {'supine': 0, 'supine_plo': 1, 'hbh': 2, 'xl': 3, 'sl': 4,
                 'phu': 5, 'lateral': 6, 'lateral_plo': 7}
    # get the file names
    new_poses = []
    for file in df['Filename']:
        if 'xl' in file:
            new_poses.append(pose_dict['xl'])
        elif 'sl' in file:
            new_poses.append(pose_dict['sl'])
        elif 'hbh' in file:
            new_poses.append(pose_dict['hbh'])
        elif 'phu' in file:
            new_poses.append(pose_dict['phu'])
        elif 'rollpi' in file:
            if 'plo' in file:
                new_poses.append(pose_dict['lateral_plo'])
            else:
                new_poses.append(pose_dict['lateral'])
        else:
            if 'plo' in file:
                new_poses.append(pose_dict['supine_plo'])
            else:
                new_poses.append(pose_dict['supine'])
    df['Pose Type'] = new_poses
    df.to_csv(pjc.TRAIN_DIR.joinpath('metadata_all_classes.csv'), index=False)
    return None
