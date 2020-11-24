# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 08:38:09 2020

@author: BTLab
"""

import bodies_at_rest as bar
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import PoseNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def train_posenet():

    # TODO: Get the paths from a seperate project constants file
    dataset = bar.BodiesAtRestDataset(
        csv_file=r'C:\Users\BTLab\Documents\Aakash\Pose Classification\Data\train\metadata.csv',
        root_dir=r'C:\Users\BTLab\Documents\Aakash\Pose Classification\Data\train')
    
    batch_size = 512
    validation_split = .1
    shuffle_dataset = True
    random_seed = 7

    # TODO: Seperate the validation data manually
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=valid_sampler, num_workers=0)
    
    device = \
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = PoseNet.create_posenet_model()
    model.to(device)
    model.train()
    
    learning_rate = 0.001
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    
    max_epochs = 1000
    num_classes = 3
    
    log_dir = Path(r"C:\Users\BTLab\Documents\Aakash\Pose Classification\LogDir")
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # TODO: Add tp, fp, tn, fn to the writer
    step = 0
    for epoch in range(0, max_epochs):
        model.train()
        metrics = \
            [{"tp": 0, "fp": 0, "tn": 0, "fn": 0} for i in range(0, num_classes)]
            
        for sample in tqdm(train_loader):
            imgs = sample['image'].to(device)
            y_pred = model(imgs)
            
            target = torch.argmax(sample['label'], dim=1)
            target = target.to(device)
    
            loss = loss_function(y_pred, target)
        
            for i in range(0, y_pred.shape[0]):
                for j in range(0, num_classes):
                    if y_pred[i][j] >= 0.5 and sample['label'][i][j] >= 0.5:
                        metrics[j]["tp"] += 1
                    elif y_pred[i][j] < 0.5 and sample['label'][i][j] >= 0.5:
                        metrics[j]["fn"] += 1
                    elif y_pred[i][j] >= 0.5 and sample['label'][i][j] < 0.5:
                        metrics[j]["fp"] += 1
                    elif y_pred[i][j] < 0.5 and sample['label'][i][j] < 0.5:
                        metrics[j]["tn"] += 1
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train', loss, step)
    
        
            for j in range(0, num_classes):
                tp = metrics[j]["tp"]
                tn = metrics[j]["tn"]
                fp = metrics[j]["fp"]
                fn = metrics[j]["fn"]
                if not (0 in [tp + tn + fp + fn, tp + fp, tp + fn, tn + fp]):
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    specificity = tn / (tn + fp)
                    writer.add_scalar(f"Train_accuracy/{j}", accuracy, step)
                    writer.add_scalar(f"Train_precision/{j}", precision, step)
                    writer.add_scalar(f"Train_recall/{j}", recall, step)
                    writer.add_scalar(f"Train_specificity/{j}", specificity, step)
            
            step += 1
    
        
            
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   str(log_dir / (str(epoch).zfill(10) + ".tar")))
    
        
        
        eval_metrics = \
            [{"tp": 0, "fp": 0, "tn": 0, "fn": 0} for i in range(0, num_classes)]
        model.eval()
        with torch.no_grad():
            for sample in valid_loader:
                imgs = sample['image'].to(device)
                y_pred = model(imgs)
                
                target = torch.argmax(sample['label'], dim=1)
                target = target.to(device)
        
                loss = loss_function(y_pred, target)
        
                for i in range(0, y_pred.shape[0]):
                    for j in range(0, num_classes):
                        if y_pred[i][j] >= 0.5 and sample['label'][i][j] >= 0.5:
                            eval_metrics[j]["tp"] += 1
                        elif y_pred[i][j] < 0.5 and sample['label'][i][j] >= 0.5:
                            eval_metrics[j]["fn"] += 1
                        elif y_pred[i][j] >= 0.5 and sample['label'][i][j] < 0.5:
                            eval_metrics[j]["fp"] += 1
                        elif y_pred[i][j] < 0.5 and sample['label'][i][j] < 0.5:
                            eval_metrics[j]["tn"] += 1
                            
        for j in range(0, num_classes):
                tp = eval_metrics[j]["tp"]
                tn = eval_metrics[j]["tn"]
                fp = eval_metrics[j]["fp"]
                fn = eval_metrics[j]["fn"]
                if not (0 in [tp + tn + fp + fn, tp + fp, tp + fn, tn + fp]):
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    specificity = tn / (tn + fp)
                    writer.add_scalar(f"Eval_accuracy/{j}", accuracy, epoch)
                    writer.add_scalar(f"Eval_precision/{j}", precision, epoch)
                    writer.add_scalar(f"Eval_recall/{j}", recall, epoch)
                    writer.add_scalar(f"Eval_specificity/{j}", specificity, epoch)
        
        

# bar.view_one_img(3, dataset)
