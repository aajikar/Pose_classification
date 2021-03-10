# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:58:32 2020

@author: BTLab
"""

from pathlib import Path

TRAIN_DIR = Path(r"C:\Users\BTLab\Documents\Aakash\Data_train_val_test\train")
TEST_DIR = Path(r"C:\Users\BTLab\Documents\Aakash\Data_train_val_test\test")
VAL_DIR = Path(r"C:\Users\BTLab\Documents\Aakash\Data_train_val_test\val")

# For three classes
md3c = 'metadata.csv'
# For eight classes
md8c = 'metadata_all_classes.csv'

TRAIN_METADATA_CSV = TRAIN_DIR.joinpath(md3c)
TRAIN_METADATA_CSV_ALL = TRAIN_DIR.joinpath(md8c)

TEST_METADATA_CSV = TEST_DIR.joinpath(md3c)
TEST_METADATA_CSV_ALL = TEST_DIR.joinpath(md8c)

VAL_METADATA_CSV = VAL_DIR.joinpath(md3c)
VAL_METADATA_CSV_ALL = VAL_DIR.joinpath(md8c)

PMAT_CSV = Path(r'C:\Users\BTLab\Documents\Aakash\PMAT\metadata.csv')
XSN_CSV = Path(r'C:\Users\BTLab\Documents\Aakash\Patient Data from Stroke Ward\Patient 0 No Thresh\metadata.csv')

X_VAL_CSV = r"C:\Users\BTLab\Documents\Aakash\Data_train_val_test\metadata_all_classes.csv"
LOG_DIR = Path(r"C:\Users\BTLab\Documents\Aakash\Pose Classification\LogDir")

BED_INC_CSV = Path(r'C:\Users\BTLab\Documents\Aakash\PMAT Experiment ii\Bed Inclination\metadata.csv')
