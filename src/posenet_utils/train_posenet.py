# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 08:38:09 2020

@author: BTLab
"""

from posenet_utils import bodies_at_rest as bar
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from posenet_utils import PoseNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import proj_constants as pjc
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from posenet_utils import training_constants as tc
import torchvision
import pandas as pd
from preprocessing.PMAT import PMAT_dataset as PMAT

def train_posenet():

    # TODO: Get the paths from a seperate project constants file
    train_dataset = bar.BodiesAtRestResnet(pjc.TRAIN_METADATA_CSV,
                                     pjc.TRAIN_DIR)
    valid_dataset = bar.BodiesAtRestResnet(pjc.VAL_METADATA_CSV, pjc.VAL_DIR)
    
    batch_size = tc.BATCH_SIZE
    validation_split = .1
    shuffle_dataset = True
    random_seed = 7
    
    TRAIN_PARAMS = {'batch size': tc.BATCH_SIZE,
                    'learning rate': tc.LEARNING_RATE,
                    'optimizer': tc.OPTIMIZER,
                    'loss': tc.LOSS,
                    'max epochs': tc.MAX_EPOCHS,
                    'num classes': tc.NUM_CLASSES}

    # # TODO: Seperate the validation data manually
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # if shuffle_dataset :
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]
    
    # # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=16)
    
    device = \
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = PoseNet.create_poseresnet_model()
    
    for param in model.parameters():
        param.requires_grad = True

    modules = list(model.modules())

    for module in modules:
        for param in module.parameters():
            if type(module) is torchvision.ops.misc.FrozenBatchNorm2d:
                param.requires_grad = False  # turn off grad on frozen backnorm layers, they were causing nan losses
    
    model.to(device)
    model.train()
    
    # TODO: Lower lr for ResNet and WideResNet
    learning_rate = tc.LEARNING_RATE
    
    optimizer = tc.OPTIMIZER(model.parameters(), lr=learning_rate)
    loss_function = tc.LOSS
    
    max_epochs = tc.MAX_EPOCHS
    num_classes = tc.NUM_CLASSES
    # num_classes = 2  # Used for temporary binary classification
    
    log_dir = Path(r"C:\Users\BTLab\Documents\Aakash\Pose Classification\LogDir")
    writer = SummaryWriter(log_dir=str(log_dir))
    
    y_true = torch.tensor([])
    all_preds = torch.tensor([])
    all_preds.to(device)
    
    # TODO: Add tp, fp, tn, fn to the writer
    step = 0
    for epoch in range(0, max_epochs):
        print(f"\nEpoch {epoch+1} of {max_epochs}\n")
        model.train()
        metrics = \
            [{"tp": 0, "fp": 0, "tn": 0, "fn": 0} for i in range(0, num_classes)]
        y_true = torch.tensor([])
        all_preds = torch.tensor([])
        print("\nTraining")
        for sample in tqdm(train_loader):
            imgs = sample['image'].to(device)
            y_pred = model(imgs)
            all_preds = torch.cat((all_preds.cpu(), y_pred.cpu()), dim=0)
            
            # Comment out when one hot encoding is not needed
            # target = torch.argmax(sample['label'], dim=1)
            target = sample['label']
            target = target.to(device)
            y_true = torch.cat((y_true.cpu(), target.cpu()), dim=0)
                     
            loss = loss_function(y_pred, target.type_as(y_pred))
        
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
            # if step != 0 and step % 289 == 0:
            #     break
                
        # # Get the classification report
        # all_preds = torch.argmax(all_preds, dim=1)
        # report = classification_report(y_true.cpu().numpy(), all_preds.cpu().numpy(), output_dict=True)
        
        
        # # Get the confusion matrix
        # cm = confusion_matrix(y_true.cpu().numpy(), all_preds.cpu().numpy(), normalize='true')
        # # Get the accuracies
        # acc = np.diag(cm)
        
        # # Write the metrics to writer
        # writer.add_scalar('loss/train', loss, epoch)
        # writer.add_scalar('global_accuracy/train', accuracy_score(y_true.cpu().numpy(), all_preds.cpu().numpy()), epoch)
        # for j in range(0, num_classes):
        #     writer.add_scalar(f"Train_accuracy/{j}", acc[j], epoch)
        # for k in report.keys():
        #     try:
        #         writer.add_scalar(f"Train_precision/{k}", report[k]['precision'], epoch)
        #         writer.add_scalar(f"Train_recall/{k}", report[k]['recall'], epoch)
        #     except:
        #         pass
        
            
        
        eval_metrics = \
            [{"tp": 0, "fp": 0, "tn": 0, "fn": 0} for i in range(0, num_classes)]
        model.eval()
        y_true = torch.tensor([])
        all_preds = torch.tensor([])
        print("\nValidating\n")
        with torch.no_grad():
            for sample in tqdm(valid_loader):
                imgs = sample['image'].to(device)
                y_pred = model(imgs)
                all_preds = torch.cat((all_preds.cpu(), y_pred.cpu()), dim=0)
                
                # target = torch.argmax(sample['label'], dim=1)
                target = sample['label']
                target = target.to(device)
                y_true = torch.cat((y_true.cpu(), target.cpu()), dim=0)
        
                loss = loss_function(y_pred, target.type_as(y_pred))
        
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
        
        # # Get the classification report
        # all_preds = torch.argmax(all_preds, dim=1)
        # report = classification_report(y_true, all_preds, output_dict=True)
        
        # # Get the confusion matrix
        # cm = confusion_matrix(y_true, all_preds, normalize='true')
        # # Get the accuracies
        # acc = np.diag(cm)
        
        # # Write the metrics to writer
        # writer.add_scalar('loss/validation', loss, epoch)
        # writer.add_scalar('global_accuracy/validation', accuracy_score(y_true.cpu().numpy(), all_preds.cpu().numpy()), epoch)
        # for j in range(0, num_classes):
        #     writer.add_scalar(f"Validation_accuracy/{j}", acc[j], epoch)
        # for k in report.keys():
        #     try:
        #         writer.add_scalar(f"Validation_precision/{int(float(k))}", report[k]['precision'], epoch)
        #         writer.add_scalar(f"Validation_recall/{int(float(k))}", report[k]['recall'], epoch)
        #     except:
        #         pass                    
        
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
                    writer.add_scalar(f"Eval_true_negative/{j}", tn, epoch)
                    writer.add_scalar(f"Eval_true_positive/{j}", tp, epoch)
                    writer.add_scalar(f"Eval_false_negative/{j}", fn, epoch)
                    writer.add_scalar(f"Eval_false_positive/{j}", fp, epoch)
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'training parameters': TRAIN_PARAMS},
               str(log_dir / (str(epoch).zfill(10) + ".tar")))
    
    return model


def calculate_metrics(true_labels, predictions, num_classes, metric):
    """
    Calculate tp, fp, fn, and tn of batch of samples.

    Parameters
    ----------
    true_labels : torch.Tensor
        Torch tensor containig the true labels.
    predictions : torch.Tensor
        Torch tensor containg the predicted labels.
    num_classes : int
        Number of classes.

    Returns
    -------
    metric : list
        List of dictionaries containing tp, fp, fn, and tn for each class.

    """
    # Go through all classes and predictions
    for i in range(0, predictions.shape[0]):
        for j in range(0, num_classes):

            # If estimate and true label are greater than 0.5 then it is a tp
            if predictions[i][j] >= 0.5 and true_labels[i][j] >= 0.5:
                metric[j]['tp'] += 1

            # If estimate is less than 0.5 and true lable is greater than 0.5
            # then this is a fn
            elif predictions[i][j] < 0.5 and true_labels[i][j] >= 0.5:
                metric[j]['fn'] += 1

            # If estimate is greater than 0.5 and true label is less than 0.5
            # then this is a fp
            elif predictions[i][j] >= 0.5 and true_labels[i][j] < 0.5:
                metric[j]['fp'] += 1

            # If estimate and true label are less than 0.5 then it is a tn
            elif predictions[i][j] < 0.5 and true_labels[i][j] < 0.5:
                metric[j]['tn'] += 1

            tp = metric[j]['tp']
            fp = metric[j]['fp']
            fn = metric[j]['fn']
            tn = metric[j]['tn']

            # Calculate accuracy, precision, recall, and specificity
            if not (0 in [tp + tn + fp + fn]):
                metric[j]['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            if not (0 in [tp + fp]):
                metric[j]['precision'] = tp / (tp + fp)
            if not (0 in [tp + fn]):
                metric[j]['recall'] = tp / (tp + fn)
            if not (0 in [tn + fp]):
                metric[j]['specificity'] = tn / (tn + fp)

    return metric


def write_metrics(metric, num_classes, writer, step, loss=None, mode='Train',
                  include_count=False):
    """
    Write calculated metrics to tensorboard.

    Parameters
    ----------
    metric : list
        List of dict containing tp, fp, tn, and fn for each class.
    num_classes : int
        Number of classes in the output.
    writer : SummaryWriter
        Tensorboard SummaryWriter object.
    step : int
        The current step for the writer.
    loss : float, optional
        Loss calculatd between predictions and true labels.
        The default is 'None'.
    mode : str, optional
        Train, eval, or test. The default is 'Train'.
    include_count : bool, optional
        Boolean to include tp, fp, tn, and fn. The default is False.

    Returns
    -------
    None.

    """
    j = 0
    for i in range(0, num_classes):
        tp = metric[i]['tp']
        fp = metric[i]['fp']
        fn = metric[i]['fn']
        tn = metric[i]['tn']

        # Write accuracy, precision, recall, and specificity
        writer.add_scalar(f"{mode}_accuracy/{j}", metric[i]['accuracy'], step)
        writer.add_scalar(f"{mode}_precision/{j}", metric[i]['precision'], step)
        writer.add_scalar(f"{mode}_recall/{j}", metric[i]['recall'], step)
        writer.add_scalar(f"{mode}_specificity/{j}", metric[i]['specificity'], step)

        if include_count:
            writer.add_scalar(f"{mode}_true_negative/{j}", metric[i]['tn'], step)
            writer.add_scalar(f"{mode}_true_positive/{j}", metric[i]['tp'], step)
            writer.add_scalar(f"{mode}_false_negative/{j}", metric[i]['fn'], step)
            writer.add_scalar(f"{mode}_false_positive/{j}", metric[i]['fp'], step)
        
        j += 1

    if mode == "Train":
        writer.add_scalar('loss/train', loss, step)

        return None


def save_model(model, optimizer, TRAIN_PARAMS, epoch, log_dir):
    """
    Save the model in the specified directory as tar file.

    Parameters
    ----------
    model : nn.Module
        Pytorch model.
    optimizer : torch.optim
        Optimizer used for training.
    TRAIN_PARAMS : dict
        Parameters used for training.
    epoch : int
        Number of elapsed epochs.
    log_dir : str
        Location of the directory where file should be stored.

    Returns
    -------
    None.

    """
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'training parameters': TRAIN_PARAMS},
               str(log_dir / (str(epoch).zfill(10) + ".tar")))

    return None


def validation_step(model, optimizer, loss_function, valid_loader, device,
                    num_classes, epoch, writer):
    """
    Perform the validation step in the epoch.

    Parameters
    ----------
    model : nn.Module
        Pytorch model.
    optimizer : torch.optim
        Optimizer used for training the model.
    loss_function : torch.loss
        Loss function used in the model.
    valid_loader : DataLoader
        DataLoader object containing the validation data.
    device : torch.device
        The device where the data is located.
    num_classes : int
        Number of outputs from the model.
    epoch : int
        Current elapsed epochs.
    writer : SummaryWriter
        Tensorboard SummaryWriter object.

    Returns
    -------
    valid_results : list
        List of dictionary containing accuracy, precision, recall,
        and specificity for each class.

    """
    # First set the model into evaluation mode
    model.eval()
    print("\nValidating\n")

    # Create metrics prior to starting validation loop
    metric = [{'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'accuracy': 0,
               'precision': 0, 'recall': 0, 'specificity': 0}
              for i in range(num_classes)]

    # Set to no grad
    with torch.no_grad():
        for sample in tqdm(valid_loader):
            imgs = sample['image'].to(device)
            y_pred = model(imgs)
            target = sample['label']
            target = target.to(device)

            # Calculate metrics
            metric = calculate_metrics(target, y_pred, num_classes, metric)

            # Write metrics
            write_metrics(metric, num_classes, writer, step=epoch, mode="Eval")

    valid_results = [{'accuracy': metric[i]['accuracy'],
                      'precision': metric[i]['precision'],
                      'recall': metric[i]['recall'],
                      'specificity': metric[i]['specificity']}
                     for i in range(num_classes)]
    return valid_results


def train_step(model, optimizer, loss_function, train_loader, device,
               num_classes, epoch, writer, step):
    """
    Perform the training step in the epoch.

    Parameters
    ----------
    model : nn.Module
        Pytorch model.
    optimizer : torch.optim
        Optimizer used for training the model.
    loss_function : torch.loss
        Loss function used in the model.
    valid_loader : DataLoader
        DataLoader object containing the validation data.
    device : torch.device
        The device where the data is located.
    num_classes : int
        Number of outputs from the model.
    epoch : int
        Current elapsed epochs.
    writer : SummaryWriter
        Tensorboard SummaryWriter object.
    step : int
        Global step of the current sample.

    Returns
    -------
    train_results : list
        List of dictionary containing accuracy, precision, recall,
        and specificity for each class.
    step : int
        Updated global step of the current sample.

    """
    # First set the model into training mode
    model.train()
    print("\nTraining\n")
    
    # Create metrics prior to starting training loop
    metric = [{'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'accuracy': 0,
               'precision': 0, 'recall': 0, 'specificity': 0}
              for i in range(num_classes)]

    for sample in tqdm(train_loader):
        imgs = sample['image'].to(device)
        y_pred = model(imgs)
        target = sample['label']
        target = target.to(device)
        loss = loss_function(y_pred, target.type_as(y_pred))

        # Calculate metrics
        metric = calculate_metrics(target, y_pred, num_classes, metric)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Write metrics
        write_metrics(metric, num_classes, writer, step, loss)

        # Increment step
        step += 1

    # Collect results for that epoch
    train_results = [{'accuracy': metric[i]['accuracy'],
                      'precision': metric[i]['precision'],
                      'recall': metric[i]['recall'],
                      'specificity': metric[i]['specificity']}
                     for i in range(num_classes)]

    return train_results, step


def cross_validate(model=None, optimizer=None, dataset=None, k_fold=5,
                   num_classes=8, verbosity=False, batch_size=tc.BATCH_SIZE,
                   max_epochs=tc.MAX_EPOCHS, log_dir=None,
                   loss_function=tc.LOSS, device=None, train_params=None):
    """
    Perform k-fold cross validation on dataset with a model.

    Parameters
    ----------
    model : nn.Module, optional
        Pytorch model. The default is None.
    optimizer : nn.optim, optional
        Optimizer used for the model. The default is None.
    dataset : DataLoader, optional
        Dataset used for training and validation. The default is None.
    k_fold : int, optional
        Number of folds for cross validation. The default is 5.
    num_classes : int, optional
        The number of classes the model outputs. The default is 8.
    verbosity : bool, optional
        Indicate level of text output. The default is False.
    batch_size : int, optional
        Batch size of samples used during training and validation.
        The default is tc.BATCH_SIZE.
    max_epochs : int, optional
        Number of epochs model will train for. The default is tc.MAX_EPOCHS.
    log_dir : pathlib.Path, optional
        Path to where the log dir for each epoch is located.
        The default is None.
    loss_function : nn.loss, optional
        Loss function used during training. The default is tc.LOSS.
    device : nn.Device, optional
        The device where training will be performed. The default is None.
    train_params : dict
        Dictionary of parameters used for training.

    Returns
    -------
    train_scores : list
        List of dictionaries for each class containing the accuracy, precision,
        recall, specificity, and F1 score for each k-fold.
    val_scores : list
        List of dictionaries for each class containing the accuracy, precision,
        recall, specificity, and F1 score for each k-fold.

    """
    train_scores = [{'Accuracy': pd.Series(),
                     'Precision': pd.Series(),
                     'Recall': pd.Series(),
                     'Specificity': pd.Series(),
                     'F1': pd.Series()}
                    for i in range(num_classes)]

    val_scores = [{'Accuracy': pd.Series(),
                   'Precision': pd.Series(),
                   'Recall': pd.Series(),
                   'Specificity': pd.Series(),
                   'F1': pd.Series()}
                  for i in range(num_classes)]
    
    total_size = len(dataset)
    
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    
    torch.cuda.empty_cache()
    
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        # Initialize the model at start of each fold
        model = PoseNet.create_poseresnet_model()
    
        for param in model.parameters():
            param.requires_grad = True
    
        modules = list(model.modules())
    
        for module in modules:
            for param in module.parameters():
                if type(module) is torchvision.ops.misc.FrozenBatchNorm2d:
                    param.requires_grad = False  # turn off grad on frozen backnorm layers, they were causing nan losses
        
        model.to(device)
        
        # Setup the optimizer
        learning_rate = tc.LEARNING_RATE
    
        optimizer = tc.OPTIMIZER(model.parameters(), lr=learning_rate)
        
        
        # Set up the folds
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        
        # Update the current log dir for the k fold
        folder_name = "Fold" + str(i)
        current_log_dir = Path.joinpath(log_dir, folder_name)
        
        # Create the tensorboard
        writer = SummaryWriter(log_dir=str(current_log_dir))
        
        if verbosity:
            print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
              % (trll,trlr,trrl,trrr,vall,valr))
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset,val_indices)
        
        if verbosity:
            print(len(train_set),len(val_set))
            print()
        
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=16)
        valid_loader = DataLoader(val_set, batch_size=batch_size,
                                  shuffle=True, num_workers=16)
        
        step = 0
        
        for epoch in range(0, max_epochs):
            print(f"\nEpoch {epoch+1} of {max_epochs}\n")

            # Perform training
            train_results, step = train_step(model, optimizer, loss_function,
                                             train_loader, device, num_classes,
                                             epoch, writer, step)
            
            # Perform validation
            valid_results = validation_step(model, optimizer, loss_function,
                                            valid_loader, device, num_classes,
                                            epoch, writer)
            
            # Increment step
            step += 1

        # Get the final scores after all the epochs
        for j in range(num_classes):
            train_scores[j]['Accuracy'].at[i] = train_results[j]['accuracy']
            train_scores[j]['Precision'].at[i] = train_results[j]['precision']
            train_scores[j]['Recall'].at[i] = train_results[j]['recall']
            train_scores[j]['Specificity'].at[i] = train_results[j]['specificity']
            if not (0 in [train_results[j]['precision'] + train_results[j]['recall']]):
                train_scores[j]['F1'].at[i] = 2 * ((train_results[j]['precision'] *
                                                    train_results[j]['recall']) / 
                                                   (train_results[j]['precision'] +
                                                    train_results[j]['recall']))
            else:
                train_scores[j]['F1'].at[i] = 0
            
            val_scores[j]['Accuracy'].at[i] = valid_results[j]['accuracy']
            val_scores[j]['Precision'].at[i] = valid_results[j]['precision']
            val_scores[j]['Recall'].at[i] = valid_results[j]['recall']
            val_scores[j]['Specificity'].at[i] = valid_results[j]['specificity']
            if not (0 in [valid_results[j]['precision'] + valid_results[j]['recall']]):
                val_scores[j]['F1'].at[i] = 2 * ((valid_results[j]['precision'] *
                                                    valid_results[j]['recall']) / 
                                                   (valid_results[j]['precision'] +
                                                    valid_results[j]['recall']))
            else:
                val_scores[j]['F1'].at[i] = 0
        
        save_model(model, optimizer, train_params, max_epochs-1, current_log_dir)
        
        torch.cuda.empty_cache()
        
    return train_scores, val_scores, model


def PMAT_data_training():
    # Load in the dataset
    torch.cuda.empty_cache()

    csv_file = pjc.PMAT_CSV
    dataset = PMAT(csv_file, use_PMAT_labels=True)

    # Setup the training parameters
    TRAIN_PARAMS = {'batch size': tc.BATCH_SIZE,
                    'learning rate': tc.LEARNING_RATE,
                    'optimizer': tc.OPTIMIZER,
                    'loss': tc.LOSS,
                    'max epochs': tc.MAX_EPOCHS,
                    'num classes': tc.NUM_CLASSES}

    device = \
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

    loss_function = tc.LOSS

    max_epochs = tc.MAX_EPOCHS
    # max_epochs = 3
    num_classes = tc.NUM_CLASSES

    log_dir = pjc.LOG_DIR
    
    batch_size = tc.BATCH_SIZE
    
    train_scores = [{'Accuracy': pd.Series(),
                     'Precision': pd.Series(),
                     'Recall': pd.Series(),
                     'Specificity': pd.Series(),
                     'F1': pd.Series()}
                    for i in range(num_classes)]

    val_scores = [{'Accuracy': pd.Series(),
                   'Precision': pd.Series(),
                   'Recall': pd.Series(),
                   'Specificity': pd.Series(),
                   'F1': pd.Series()}
                  for i in range(num_classes)]

    # Then perform Leave One Subject Out LOSO training
    # There are 13 subjects so 13 times training
    for sub in range(13):
        # Create a new dataframe that doesn't contain sub+1 subject
        # Have a training set and validation set
        # Subject ID index starts at 1
        train_indices, val_indices = dataset.get_LOSO_split(sub+1)
        
        # Next make the folder name for each LOSO
        folder_name = "LOSO" + str(sub+1)
        current_log_dir = Path.joinpath(log_dir, folder_name)
        
        # Create the tensorboard
        writer = SummaryWriter(log_dir=str(current_log_dir))
        
        # Create Dataset out of the dataframes
        train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=16)
        valid_loader = DataLoader(val_set, batch_size=batch_size,
                                  shuffle=True, num_workers=16)
        
        step = 0
        
        # Next create the model
        model = PoseNet.create_poseresnet_model()
    
        for param in model.parameters():
            param.requires_grad = True
    
        modules = list(model.modules())
    
        for module in modules:
            for param in module.parameters():
                if type(module) is torchvision.ops.misc.FrozenBatchNorm2d:
                    param.requires_grad = False  # turn off grad on frozen backnorm layers, they were causing nan losses
        
        model.to(device)
        
        # Setup the optimizer
        learning_rate = tc.LEARNING_RATE
    
        optimizer = tc.OPTIMIZER(model.parameters(), lr=learning_rate)
        
        for epoch in range(0, max_epochs):
            print(f"\nEpoch {epoch+1} of {max_epochs}\n")

            # Perform training
            train_results, step = train_step(model, optimizer, loss_function,
                                             train_loader, device, num_classes,
                                             epoch, writer, step)
            
            # Perform validation
            valid_results = validation_step(model, optimizer, loss_function,
                                            valid_loader, device, num_classes,
                                            epoch, writer)
            
            # Increment step
            step += 1
        
        # Get the final scores after all the epochs
        for j in range(num_classes):
            train_scores[j]['Accuracy'].at[sub] = train_results[j]['accuracy']
            train_scores[j]['Precision'].at[sub] = train_results[j]['precision']
            train_scores[j]['Recall'].at[sub] = train_results[j]['recall']
            train_scores[j]['Specificity'].at[sub] = train_results[j]['specificity']
            if not (0 in [train_results[j]['precision'] + train_results[j]['recall']]):
                train_scores[j]['F1'].at[sub] = 2 * ((train_results[j]['precision'] *
                                                    train_results[j]['recall']) / 
                                                   (train_results[j]['precision'] +
                                                    train_results[j]['recall']))
            else:
                train_scores[j]['F1'].at[sub] = 0
            
            val_scores[j]['Accuracy'].at[sub] = valid_results[j]['accuracy']
            val_scores[j]['Precision'].at[sub] = valid_results[j]['precision']
            val_scores[j]['Recall'].at[sub] = valid_results[j]['recall']
            val_scores[j]['Specificity'].at[sub] = valid_results[j]['specificity']
            if not (0 in [valid_results[j]['precision'] + valid_results[j]['recall']]):
                val_scores[j]['F1'].at[sub] = 2 * ((valid_results[j]['precision'] *
                                                    valid_results[j]['recall']) / 
                                                   (valid_results[j]['precision'] +
                                                    valid_results[j]['recall']))
            else:
                val_scores[j]['F1'].at[sub] = 0
        
        save_model(model, optimizer, TRAIN_PARAMS, max_epochs-1, current_log_dir)
        
        torch.cuda.empty_cache()
        
    return model, train_scores, val_scores
        
    

def train_model(x_validate=False):
    torch.cuda.empty_cache()
    
    csv_file = pjc.X_VAL_CSV
    dataset = bar.BodiesAtRestMultilableResNet(csv_file, pjc.TRAIN_DIR)
    
    TRAIN_PARAMS = {'batch size': tc.BATCH_SIZE,
                    'learning rate': tc.LEARNING_RATE,
                    'optimizer': tc.OPTIMIZER,
                    'loss': tc.LOSS,
                    'max epochs': tc.MAX_EPOCHS,
                    'num classes': tc.NUM_CLASSES}
    
    device = \
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    loss_function = tc.LOSS
    
    max_epochs = tc.MAX_EPOCHS
    # max_epochs = 3
    num_classes = tc.NUM_CLASSES
    
    log_dir = pjc.LOG_DIR
    
    if x_validate:
        train_scores, val_scores, model = cross_validate(dataset=dataset,
                                                         max_epochs=max_epochs,
                                                         device=device,
                                                         train_params=TRAIN_PARAMS,
                                                         log_dir=log_dir,
                                                         verbosity=False,
                                                         k_fold=5)
    
        return model, train_scores, val_scores
    
    else:
        model = train_posenet()
        return model
    
    
# bar.view_one_img(3, dataset)
