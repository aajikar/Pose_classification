# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:04:35 2020

@author: BTLab
"""

import torch

BATCH_SIZE = 128
LEARNING_RATE = 0.0001

OPTIMIZER = torch.optim.Adam
LOSS = torch.nn.BCEWithLogitsLoss()

MAX_EPOCHS = 10
NUM_CLASSES = 3
