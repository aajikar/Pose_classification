# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:57:26 2020.

@author: BTLab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet
import numpy as np


class Net(nn.Module):
    """Class for testing how the nn.Module works."""

    def __init__(self):
        """
        Initialize the function.

        There are four convolutional layers followed by three fully connnected
        layers in the function.

        Returns
        -------
        None.

        """
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
        """
        Peform the convolutional part of the network.

        Parameters
        ----------
        x : tensor
            Four dimensional tensor in the shape
            (batch, channel, height, width).

        Returns
        -------
        x : tensor
            Transformed four dimensional tensor in the shape
            (batch, channel, height, width).

        """
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
        """
        Forward function of a nerual network.

        Parameters
        ----------
        x : tensor
            Four dimensional tensor in the shape
            (batch, channel, height, width).

        Returns
        -------
        x : tensor
            Predicted output from the forward function. Has a softmax applied
            to it over the predictions in one sample.

        """
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class input_layer(nn.Module):
    """Input Layer of conv network to mimic Bodies at rest conv layer."""

    def __init__(self):
        """
        Initialize the function with convolutional layer of size 7x7.

        The convolutional layer has a stride of 2 and padding of 3, and outputs
        192 channels.

        Returns
        -------
        None.

        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 192, 7, stride=2, padding=3)

    def forward(self, x):
        """
        Forward function of the neural net.

        The output has a dropout layer.

        Parameters
        ----------
        x : tensor
            Four dimensional tensor in the shape
            (batch, channel, height, width).

        Returns
        -------
        x : tensor
            Transformed four dimensional tensor in the shape
            (batch, channel, height, width).

        """
        x = F.tanh(self.conv1(x))
        x = F.dropout(x, p=0.1)
        return x


class InputBatch(nn.Module):
    """
    Input Layer to mimic Bodies at Rest input convolutional layer.

    The input layer has convolutional filter of size 7x7, with a stride of 2,
    a padding of 3, and outputs 192 features. The dropout layer is replaced by
    a batchnorm layer.

    """

    def __init__(self):
        """
        Intialize the function.

        Returns
        -------
        None.

        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 192, 7, stride=2, padding=3)
        self.bn_0 = torch.nn.BatchNorm2d(192)

    def forward(self, x):
        """
        Forward function of the neural network.

        Parameters
        ----------
        x : tensor
            Four dimensional tensor in the shape
            (batch, channel, height, width).

        Returns
        -------
        x : tensor
            Transformed four dimensional tensor in the shape
            (batch, channel, height, width).

        """
        x = F.tanh(self.conv1(x))
        x = self.bn_0(x)
        return x


class backbone(nn.Module):
    """
    The backbone of the CNN for classification based on bodies at rest.

    The backbone contains dropout layers of stregth 0.1 between the different
    conv layers.

    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(192, 192, 3)
        self.conv2 = nn.Conv2d(192, 384, 3)
        self.conv3 = nn.Conv2d(384, 192, 3)

        self.to_linear = None

    def forward(self, x):
        """
        Forward method of a neural network.

        The forward function converts the input 2D image to a 1D tensor at the
        end. The size of the outgoing vector is the channels * height * width.

        Parameters
        ----------
        x : TYPE
            Four dimensional tensor in the shape
            (batch, channel, height, width).

        Returns
        -------
        x : tensor
            Transformed two dimensional tensor in the shape
            (samples, features).

        """
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
    """
    Backbone layer of the CNN based on bodies at rest.

    The layers now have a batchnorm layer rather than dropout layers between
    them.

    """

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
        """
        Forward method of a neural network.

        The forward function converts the input 2D image to a 1D tensor at the
        end. The size of the outgoing vector is the channels * height * width.

        Parameters
        ----------
        x : TYPE
            Four dimensional tensor in the shape
            (batch, channel, height, width).

        Returns
        -------
        x : tensor
            Transformed two dimensional tensor in the shape
            (samples, features).

        """
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
    """
    Class with the fully connected layers to perform classifcation.

    Has four fully connected layers. The first three fully connected layers
    have default sizes of 88, 20, and 512 respectively. The last fully
    connected layer is the classification layer with default size of 8.

    """

    # TODO: dynamically calculate input size to first fc layer
    def __init__(self, fc1_size=88, fc2_size=20, fc3_size=512, fc4_size=8):
        super().__init__()
        self.fc1 = nn.Linear(39936, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)

        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)

        self.pool1 = nn.AdaptiveAvgPool2d((26, 8))

    def forward(self, x, input_size=None):
        """
        Forward function of the neural network.

        Always expects size of 192 * 26 * 8 going into the model. Between the
        first and second, second and third fully connected layers there are
        dropout layers of strength 0.5

        Each fully connected layer has a relu activation before it.

        Parameters
        ----------
        x : tensor
            Input tensor has to be of shape (samples, 192*26*8).
        input_size : tensor, optional
            1D tensor used for debugging only. Specifies the size of the input
            coming into the fully connected layers. The default is None.

        Returns
        -------
        x : tensor
            Resulting prediction from the model. If the model is in training
            the values are returned as is. If the model is not in training a
            sigmoid function is applied to the output tensor.

        """
        x = self.pool1(x)
        x = x.view(-1, 192*26*8)
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if self.training:
            return x
        else:
            return torch.sigmoid(x)


class resnet_classification_head(nn.Module):
    """
    Classification head when using variants of ResNet as backbone.

    Features four fully connected layers of default sizes 256, 128, 156, 8
    respectively.

    """

    def __init__(self, fc1_size=256, fc2_size=128, fc3_size=256, fc4_size=8):
        super().__init__()
        self.fc1 = nn.Linear(1000, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)

        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward function of the neural network.

        Always expects a tensor of (samples, 1000) as the input.

        Parameters
        ----------
        x : tensor
            Input tensor has to be of shape (samples, 192*26*8).
        input_size : tensor, optional
            1D tensor used for debugging only. Specifies the size of the input
            coming into the fully connected layers. The default is None.

        Returns
        -------
        x : tensor
            Resulting prediction from the model. If the model is in training
            the values are returned as is. If the model is not in training a
            sigmoid function is applied to the output tensor.

        """
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if self.training:
            return x
        else:
            return torch.sigmoid(x)


class pose_nn(nn.Module):
    """Neural net based on custom made backbone and classification head."""

    def __init__(self, input_layer, backbone, classification_head):
        super().__init__()
        self.input_layer = input_layer
        self.backbone = backbone
        self.classification_head = classification_head

    def forward(self, x):
        """
        Forward method of neural net.

        Parameters
        ----------
        x : tensor
            Input tensor of shape (samples, channels, height, width).

        Returns
        -------
        labels : tensor
            Output labels predicted from the model.

        """
        input_x = self.input_layer(x)
        features, out_size = self.backbone(input_x)
        labels = self.classification_head(features, out_size)
        return labels


class pose_resnet(nn.Module):
    """
    Neural network for ResNet as backbone.

    The corresponding classificatio head is used.

    """

    def __init__(self, backbone, classification_head):
        super().__init__()
        self.backbone = backbone
        self.classification_head = classification_head

    def forward(self, x):
        """
        Forward method of neural network.

        Parameters
        ----------
        x : tensor
            Input tensor of shape (samples, channels, height, width).

        Returns
        -------
        labels : tensor
            Output labels predicted from the model.

        """
        input_x = self.backbone(x)
        labels = self.classification_head(input_x)
        return labels


class single_classification_head(nn.Module):
    def __init__(self, fc1_size=8):
        super().__init__()
        self.fc1 = nn.Linear(1000, fc1_size)
    
    def forward(self, x):
        x = self.fc1(x)
        if self.training:
            return x
        else:
            return torch.sigmoid(x)


class two_layer_classification(nn.Module):
    def __init__(self, fc1_size=256, fc2_size=8):
        super().__init__()
        self.fc1 = nn.Linear(1000, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.training:
            return x
        else:
            return torch.sigmoid(x)
        

class three_layer_classification(nn.Module):
    def __init__(self, fc1_size=256, fc2_size=128, fc3_size=8):
        super().__init__()
        self.fc1 = nn.Linear(1000, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        
        self.d1 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.training:
            return x
        else:
            return torch.sigmoid(x)

        
class five_layer_classification(nn.Module):
    def __init__(self, fc1_size=256, fc2_size=128, fc3_size=256, fc4_size=128,
                 fc5_size=8):
        super().__init__()
        self.fc1 = nn.Linear(1000, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.fc5 = nn.Linear(fc4_size, fc5_size)
        
        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)
        self.d3 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))
        x = self.d3(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        if self.training:
            return x
        else:
            return torch.sigmoid(x)


class six_layer_classification(nn.Module):
    def __init__(self, fc1_size=256, fc2_size=128, fc3_size=256, fc4_size=128,
                 fc5_size=256, fc6_size=8):
        super().__init__()
        self.fc1 = nn.Linear(1000, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.fc5 = nn.Linear(fc4_size, fc5_size)
        self.fc6 = nn.Linear(fc5_size, fc6_size)
        
        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)
        self.d3 = nn.Dropout(p=0.5)
        self.d4 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))
        x = self.d3(x)
        x = F.relu(self.fc4(x))
        x = self.d4(x)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        if self.training:
            return x
        else:
            return torch.sigmoid(x)
        

class seven_layer_classification(nn.Module):
    def __init__(self, fc1_size=256, fc2_size=128, fc3_size=256, fc4_size=128,
                 fc5_size=256, fc6_size=128, fc7_size=8):
        super().__init__()
        self.fc1 = nn.Linear(1000, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.fc5 = nn.Linear(fc4_size, fc5_size)
        self.fc6 = nn.Linear(fc5_size, fc6_size)
        self.fc7 = nn.Linear(fc6_size, fc7_size)
        
        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)
        self.d3 = nn.Dropout(p=0.5)
        self.d4 = nn.Dropout(p=0.5)
        self.d5 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))
        x = self.d3(x)
        x = F.relu(self.fc4(x))
        x = self.d4(x)
        x = F.relu(self.fc5(x))
        x = self.d5(x)
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        if self.training:
            return x
        else:
            return torch.sigmoid(x)
        

class sixteen_layer_classification(nn.Module):
    def __init__(self, fc1_size=256, fc2_size=128, fc3_size=256, fc4_size=128,
                 fc5_size=256, fc6_size=128, fc7_size=256, fc8_size=128,
                 fc9_size=256, fc10_size=128, fc11_size=256, fc12_size=128,
                 fc13_size=256, fc14_size=128, fc15_size=256, fc16_size=8):
        super().__init__()
        self.fc1 = nn.Linear(1000, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.fc5 = nn.Linear(fc4_size, fc5_size)
        self.fc6 = nn.Linear(fc5_size, fc6_size)
        self.fc7 = nn.Linear(fc6_size, fc7_size)
        self.fc8 = nn.Linear(fc7_size, fc8_size)
        self.fc9 = nn.Linear(fc8_size, fc9_size)
        self.fc10 = nn.Linear(fc9_size, fc10_size)
        self.fc11 = nn.Linear(fc10_size, fc11_size)
        self.fc12 = nn.Linear(fc11_size, fc12_size)
        self.fc13 = nn.Linear(fc12_size, fc13_size)
        self.fc14 = nn.Linear(fc13_size, fc14_size)
        self.fc15 = nn.Linear(fc14_size, fc15_size)
        self.fc16 = nn.Linear(fc15_size, fc16_size)
        
        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)
        self.d3 = nn.Dropout(p=0.5)
        self.d4 = nn.Dropout(p=0.5)
        self.d5 = nn.Dropout(p=0.5)
        self.d6 = nn.Dropout(p=0.5)
        self.d7 = nn.Dropout(p=0.5)
        self.d8 = nn.Dropout(p=0.5)
        self.d9 = nn.Dropout(p=0.5)
        self.d10 = nn.Dropout(p=0.5)
        self.d11 = nn.Dropout(p=0.5)
        self.d12 = nn.Dropout(p=0.5)
        self.d13 = nn.Dropout(p=0.5)
        self.d14 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))
        x = self.d3(x)
        x = F.relu(self.fc4(x))
        x = self.d4(x)
        x = F.relu(self.fc5(x))
        x = self.d5(x)
        x = F.relu(self.fc6(x))
        x = self.d6(x)
        x = F.relu(self.fc7(x))
        x = self.d7(x)
        x = F.relu(self.fc8(x))
        x = self.d8(x)
        x = F.relu(self.fc9(x))
        x = self.d9(x)
        x = F.relu(self.fc10(x))
        x = self.d10(x)
        x = F.relu(self.fc11(x))
        x = self.d11(x)
        x = F.relu(self.fc12(x))
        x = self.d12(x)
        x = F.relu(self.fc13(x))
        x = self.d13(x)
        x = F.relu(self.fc14(x))
        x = self.d14(x)
        x = F.relu(self.fc15(x))
        x = self.fc16(x)
        if self.training:
            return x
        else:
            return torch.sigmoid(x)


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    Args_ :
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes_ :
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone, return_layers, in_channels_list, out_channels,
                 extra_blocks=None):
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone,
                                            return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class FPNClassificationHead(nn.Module):
    def __init__(self, num_classes, pool_sizes,
                 fpn_filters=256, hidden_layer_sizes=[512, 256, 512, 256],
                 keys=['0', '1', '2', '3', 'pool']):

        super(FPNClassificationHead, self).__init__()
        
        self.keys = keys
        self.avg_pool = nn.ModuleDict({key: nn.AdaptiveAvgPool2d(pool_sizes[key]) for key in self.keys})
        
        num_features = 0
        for pool_size in pool_sizes.values():
            area = pool_size[0] * pool_size[1]
            num_features += area * fpn_filters
        
        self.num_features = num_features
        
        self.fc0 = nn.Linear(self.num_features, hidden_layer_sizes[0])
        self.fc0dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.fc1dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
        self.fc2dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_layer_sizes[2], hidden_layer_sizes[3])
        
        self.fc_classifier = nn.Linear(hidden_layer_sizes[3], num_classes)

    def forward(self, x):
        pool_outs = [self.avg_pool[key](x[key])
                     for key in self.keys]

        pool_outs = [pool_out.flatten(start_dim=1) for pool_out in pool_outs]

        concat = torch.cat(tuple(pool_outs), 1)
        
        fc0_out = F.relu(self.fc0(concat))
        fc0drop_out = self.fc0dropout(fc0_out)
        
        fc1_out = F.relu(self.fc1(fc0drop_out))
        fc1drop_out = self.fc1dropout(fc1_out)
        
        fc2_out = F.relu(self.fc2(fc1drop_out))
        fc2drop_out = self.fc2dropout(fc2_out)
        
        fc3_out = F.relu(self.fc3(fc2drop_out))
        
        out = torch.sigmoid(self.fc_classifier(fc3_out))
        
        return out


def resnet_fpn_backbone(
    backbone_name,
    pretrained,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None
):
    """
    Constructs a specified ResNet backbone with FPN on top.

    Freezes the specified number of layers in the backbone.
    Examples::
        >>> from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        >>> backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
        >>> # get some dummy image
        >>> x = torch.rand(1,3,64,64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 256, 16, 16])),
        >>>    ('1', torch.Size([1, 256, 8, 8])),
        >>>    ('2', torch.Size([1, 256, 4, 4])),
        >>>    ('3', torch.Size([1, 256, 2, 2])),
        >>>    ('pool', torch.Size([1, 256, 1, 1]))]
    Args:
        backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)

    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 64
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


def create_posenet_model():
    # TODO: Change the input parameters so the model can be changed from the
    # training loop
    in_layer = InputBatch()
    backbone_layer = BackboneBatch()
    classification = classification_head(256, 128, 256, 3)
    model = pose_nn(in_layer, backbone_layer, classification)
    return model


def create_poseresnet_model(enable_binary=False):
    # TODO: Test trained vs. pretrained
    # training loop
    backbone_layer = models.wide_resnet101_2(pretrained=True)
    classification = five_layer_classification(512,256,512,256,3)
    
    if enable_binary:
        classification = five_layer_classification(512,256,512,256,2)

    model = pose_resnet(backbone_layer, classification)
    return model


def create_poseresnetfpn_model():
    # TODO: Test trained vs. pretrained
    # training loop
    backbone_layer = resnet_fpn_backbone('wide_resnet101_2', 
                                         pretrained=True, 
                                         trainable_layers=5)
    classification = FPNClassificationHead(8, pool_sizes={'0': (16, 7),
                                                          '1': (8, 4),
                                                          '2': (4, 2),
                                                          '3': (2, 1),
                                                          'pool': (1, 1)},
                                           fpn_filters=64)
    model = pose_resnet(backbone_layer, classification)
    return model


def trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


# Code to test different things
if __name__ == '__main__':
    foo = create_poseresnetfpn_model()
    x = torch.rand((1, 3, 64, 27))
    bar = foo(x)
