

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torchvision import models


def weights_init_kaiming(m):
    # https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch/blob/master/main.py

    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def fc_init_weights(m):
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch/49433937#49433937
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)





class ResNet18(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(512, dim)
        self.FC.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.FC(x)

        e = F.normalize(x)

        return e


class ResNet50(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, dim)
        self.FC.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.FC(x)

        e = F.normalize(x)

        return e


class ResNet18_cls(nn.Module):
    def __init__(self, clsNum, dim = 128):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC1 = nn.Linear(512, dim)
        self.FC2 = nn.Linear(dim, clsNum)

        self.FC1.apply(fc_init_weights)
        self.FC2.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.FC1(x)
    
        e = F.normalize(x)

        logits = self.FC2(x)

        return e, logits




class ResNet50_cls(nn.Module):
    def __init__(self, clsNum, dim = 128):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC1 = nn.Linear(2048, dim)
        self.FC2 = nn.Linear(dim, clsNum)

        self.FC1.apply(fc_init_weights)
        self.FC2.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.FC1(x)
    
        e = F.normalize(x)

        logits = self.FC2(x)

        return e, logits


