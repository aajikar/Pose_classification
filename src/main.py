# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:38:40 2020

@author: BTLab
"""

from posenet_utils.test_posenet import test_model
from posenet_utils.train_posenet import train_posenet
from posenet_utils.train_posenet import train_model, PMAT_data_training
from posenet_utils import pose_type_change as ptc
from posenet_utils.test_posenet import plot_confusion_matrix
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def convert_to_one_hot(predictions):
    y = np.empty_like(predictions)
    
    for sample in range(predictions.shape[0]):
        for index in range(predictions.shape[1]):
            if predictions[sample][index] >= 0.5:
                y[sample][index] = 1
            else:
                y[sample][index] = 0
    
    return y


def make_multiple_conf_mat(label, y_true, y_pred):
    conf_mat_dict={}
    for label_col in range(len(label)):
        y_true_label = y_true[:, label_col]
        y_pred_label = y_pred[:, label_col]
        conf_mat_dict[label[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)
    
    return conf_mat_dict


def save_conf_mats(classes, conf_mat_dict):
    for c in classes:
        for label, matrix in conf_mat_dict.items():
            # print("Confusion matrix for label {}:".format(label))
            # print(matrix)
            plt.figure()
            if c == 'False':
                plot_confusion_matrix(np.rot90(np.rot90(matrix)), classes, title=label, normalize=False)
                if not os.path.exists(r"C:\Users\BTLab\Documents\Aakash\Pose Classification\Temp_Plots\NotNormal"):
                    os.mkdir(r"C:\Users\BTLab\Documents\Aakash\Pose Classification\Temp_Plots\NotNormal")
                plt.savefig(r"C:\Users\BTLab\Documents\Aakash\Pose Classification\Temp_Plots\NotNormal\{}.pdf".format(label))
            else:
                plot_confusion_matrix(np.rot90(np.rot90(matrix)), classes, title=label, normalize=True)
                if not os.path.exists(r"C:\Users\BTLab\Documents\Aakash\Pose Classification\Temp_Plots"):
                    os.mkdir(r"C:\Users\BTLab\Documents\Aakash\Pose Classification\Temp_Plots")
                plt.savefig(r"C:\Users\BTLab\Documents\Aakash\Pose Classification\Temp_Plots\{}.pdf".format(label))
                
    return None


# ptc.convert_to_all_poses()
if __name__ == '__main__':
    # Cross validation code
    start_time = time.time()
    # trained_model, train_scores, val_scores = train_model(x_validate=False)
    # trained_model = train_model(x_validate=False)
    import torch
    from posenet_utils import PoseNet
    model = PoseNet.create_poseresnet_model()
    model.load_state_dict(torch.load(r'C:\Users\BTLab\Documents\Aakash\Pose Classification\LogDir\0000000009.tar')['model'])
    preds, y_true, metrics = test_model(model)
    # y_preds, y_true, test_metrics = test_model(trained_model)
    # y_pred = convert_to_one_hot(y_preds)
    # labels = ['supine', 'supine_plo', 'hbh', 'xl',
    #           'sl', 'phu', 'lateral', 'lateral_plo']
    # # For 3 classes only
    # labels = ['supine', 'lateral', 'prone']
    # conf_mat_dict = make_multiple_conf_mat(labels, y_true, y_pred)
    # classes = ['True', 'False']
    # save_conf_mats(classes, conf_mat_dict)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Code for testing PMAT
    # import torch
    # from posenet_utils import PoseNet
    # from posenet_utils.test_posenet import test_model
    # model = PoseNet.create_poseresnet_model()
    # model.load_state_dict(torch.load(r'C:\Users\BTLab\Documents\Aakash\Pose Classification\Saved Models\3 Class 10 Epoch Resnet101 5 Dense Layers (256-512)\0000000009.tar')['model'])
    # y_preds, y_true, test_metrics = test_model(model)


# for i in range(len(val_scores)):
#     print(f'Average recall for class {i} is: {pd.DataFrame.mean(val_scores[i]["Recall"])}')
#     print(f'Std of recall for class {i} is: {pd.DataFrame.std(val_scores[i]["Recall"])}')yp
#     print(f'Average precision for class {i} is: {pd.DataFrame.mean(val_scores[i]["Precision"])}')
#     print(f'Std of precision for class {i} is: {pd.DataFrame.std(val_scores[i]["Precision"])}')
#     print(f'Average F1 Score for class {i} is: {pd.DataFrame.mean(val_scores[i]["F1"])}')
#     print(f'Std of F1 Score for class {i} is: {pd.DataFrame.std(val_scores[i]["F1"])}')
#     print(f'Average accuracy for class {i} is: {pd.DataFrame.mean(val_scores[i]["Accuracy"])}')
#     print(f'Std of accuracy for class {i} is: {pd.DataFrame.std(val_scores[i]["Accuracy"])}')
#     print()
=======
if __name__ == "__main__":
	train_posenet()
>>>>>>> 7444dbbd90d17a16d3103033d7c5048648558e6e
