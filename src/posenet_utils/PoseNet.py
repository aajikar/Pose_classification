# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:57:26 2020

@author: BTLab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 192, 7, 2, 3)
        self.conv2 = nn.Conv2d(192, 192, 3)
        self.conv3 = nn.Conv2d(192, 384, 3)
        self.conv4 = nn.Conv2d(384, 384, 3)

        x = torch.rand(64, 27).view(-1, 1, 64, 27)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 88)
        self.fc2 = nn.Linear(88, 20)
        self.fc3 = nn.Linear(20, 3)
        
    def convs(self, x):
        print(x[0].shape)
        x = F.tanh(self.conv1(x))
        x = F.dropout(x, p=0.1)
        print(x[0].shape)
        x = F.tanh(self.conv2(x))
        x = F.dropout(x, p=0.1)
        print(x[0].shape)
        x = F.tanh(self.conv3(x))
        print(x[0].shape)
        x = F.dropout(x, p=0.1)
        x = F.tanh(self.conv4(x))
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        print(self._to_linear)
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class input_layer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 192, 7,stride=2,padding=3)
        
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.dropout(x, p=0.1)
        return x


class InputBatch(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 192, 7,stride=2,padding=3)
        self.bn_0 = torch.nn.BatchNorm2d(192)
        
        
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.bn_0(x)
        return x

class backbone(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(192, 192, 3)
        self.conv2 = nn.Conv2d(192, 384, 3)
        self.conv3 = nn.Conv2d(384, 192, 3)
        
        self.to_linear = None
    
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.dropout(x, p=0.1)
        x = F.tanh(self.conv2(x))
        x = F.dropout(x, p=0.1)
        x = F.tanh(self.conv3(x))
        x = F.dropout(x, p=0.1)
        
        if self.to_linear is None:
            self.to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        
        return x, self.to_linear
    

class BackboneBatch(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(192, 192, 3)
        self.conv2 = nn.Conv2d(192, 384, 3)
        self.conv3 = nn.Conv2d(384, 192, 3)
        
        self.bn_0 = torch.nn.BatchNorm2d(192)
        self.bn_1 = torch.nn.BatchNorm2d(384)
        self.bn_2 = torch.nn.BatchNorm2d(192)
      
        self.to_linear = None
    
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.bn_0(x)
        x = F.tanh(self.conv2(x))
        x = self.bn_1(x)
        x = F.tanh(self.conv3(x))
        x = self.bn_2(x)
        
        if self.to_linear is None:
            self.to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        
        return x, self.to_linear


class classification_head(nn.Module):
    # TODO: dynamically calculate input size to first fc layer
    def __init__(self, fc1_size=88, fc2_size=20, fc3_size=512, fc4_size=3):
        super().__init__()
        self.fc1 = nn.Linear(39936, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        
        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)

    def forward(self, x, input_size):
        x = x.view(-1, input_size)
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if self.training:
            return x
        else:
            return F.softmax(x, dim=1)
    


class pose_nn(nn.Module):
    
    def __init__(self, input_layer, backbone, classification_head):
        super().__init__()
        self.input_layer = input_layer
        self.backbone = backbone
        self.classification_head = classification_head
    
    def forward(self, x):
        input_x = self.input_layer(x)
        features, out_size = self.backbone(input_x)
        labels = self.classification_head(features, out_size)
        return labels




def create_posenet_model():
    in_layer = InputBatch()
    backbone_layer = BackboneBatch()
    classification = classification_head(256, 128, 256, 3)
    model = pose_nn(in_layer, backbone_layer, classification)
    return model

# foo = create_posenet_model()
