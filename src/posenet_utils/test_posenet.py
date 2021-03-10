# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 08:25:33 2020

@author: BTLab
"""

import torch
import proj_constants as pjc
from posenet_utils import bodies_at_rest as bar
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from posenet_utils import PoseNet
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from preprocessing.PMAT import BedInclination as PMAT
from preprocessing.PMAT import transform
from preprocessing.XSN import XSNDataset as XSN
from torchvision import transforms

def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds


def test_model(model):
    dataset = XSN(pjc.XSN_CSV, transform)
    batch_size = 128
    shuffle_data = True
    num_classes = 3
    loss_function = torch.nn.BCEWithLogitsLoss()
    test_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=shuffle_data, num_workers=16)
    device = \
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    y_true = torch.tensor([])
    all_preds = torch.tensor([])
    all_preds.to(device)
    test_metrics = \
            [{"tp": 0, "fp": 0, "tn": 0, "fn": 0} for i in range(0, num_classes)]
    model.eval()
    print("Testing")
    with torch.no_grad():
        for sample in tqdm(test_loader):
            
            imgs = sample['image'].to(device)
            y_pred = model(imgs)
            all_preds = torch.cat((all_preds.cpu(), y_pred.cpu()), dim=0)
            
            # target = torch.argmax(sample['label'], dim=1)
            target = sample['label']
            target = target.to(device)
            y_true = torch.cat((y_true.cpu(), target.cpu()), dim=0)
            
            # loss = loss_function(y_pred, target.type_as(y_pred))
            
            for i in range(0, y_pred.shape[0]):
                for j in range(0, num_classes):
                    if y_pred[i][j] >= 0.5 and sample['label'][i][j] >= 0.5:
                        test_metrics[j]["tp"] += 1
                    elif y_pred[i][j] < 0.5 and sample['label'][i][j] >= 0.5:
                        test_metrics[j]["fn"] += 1
                    elif y_pred[i][j] >= 0.5 and sample['label'][i][j] < 0.5:
                        test_metrics[j]["fp"] += 1
                    elif y_pred[i][j] < 0.5 and sample['label'][i][j] < 0.5:
                        test_metrics[j]["tn"] += 1

        for j in range(0, num_classes):
            tp = test_metrics[j]["tp"]
            tn = test_metrics[j]["tn"]
            fp = test_metrics[j]["fp"]
            fn = test_metrics[j]["fn"]
            if not (0 in [tp + tn + fp + fn, tp + fp, tp + fn, tn + fp]):
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                specificity = tn / (tn + fp)
                print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nSpecificity: {specificity}")
            
        # all_preds = torch.argmax(all_preds, dim=1)
        # cnf_mat = multilabel_confusion_matrix(y_true.cpu().numpy(), all_preds.cpu().numpy())

        
    
    return all_preds.cpu(), y_true.cpu(), test_metrics


def load_model(filename):
    model = PoseNet.create_posenet_model()
    model.load_state_dict(torch.load(filename))
    return model


def test_pose_net():
    model = load_model(r"C:\Users\BTLab\Documents\Aakash\Pose Classification\ThreeClass1000Epoch\0000000020.tar")
    cnf_mat = test_model(model)
    print(cnf_mat)
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=14, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(r"C:\Users\BTLab\Documents\Aakash\Pose Classification\Plots\Conf_mat_8_class_flip_iter.pdf")