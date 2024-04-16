# -*- coding=utf-8 -*-

import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import math
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F


class SCNResnet(nn.Module):
    def __init__(self, model, feature_dim=512, num_classes=7):
        super(SCNResnet, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,

        )

        self.fc = nn.Linear(self.feature_dim, self.num_classes)

    def forward(self, feature):
        feature = self.features(feature)
        feature = F.adaptive_avg_pool2d(feature, (1, 1))
        feature = feature.view(feature.size(0), -1)
        x = self.fc(feature)

        return x


def pure_resnet18(feature_dim=512, num_classes=7, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    return SCNResnet(model, feature_dim=feature_dim, num_classes=num_classes)


def res18_vggface(feature_dim=512, num_classes=7):
    model = models.resnet18()
    checkpoint = torch.load('/home/seven/tools/ijba_res18_naive.pth.tar')
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
    for key in pretrained_state_dict:
        # print(key)
        # if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias') | (key == 'module.feature.weight') | (
                key == 'module.feature.bias')):
            pass
        else:
            print(key)
            model_state_dict[key] = pretrained_state_dict[key]
    model.load_state_dict(model_state_dict, strict=False)
    return SCNResnet(model, feature_dim=feature_dim, num_classes=num_classes)
