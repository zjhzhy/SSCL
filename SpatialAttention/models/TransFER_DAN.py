# -*- coding=utf-8 -*-
from SpatialAttention.models.TransFormer import Transformer, Attention, PreViT384
from SpatialAttention.models.resnet import BasicBlock
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import random
from itertools import repeat


class LANet(nn.Module):

    def __init__(self, in_dim, r=0.5):
        super(LANet, self).__init__()
        self.in_dim = in_dim
        self.mid_dim = int(in_dim * r)
        self.conv1 = nn.Conv2d(kernel_size=(1, 1), in_channels=self.in_dim, out_channels=self.mid_dim)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(kernel_size=(1, 1), in_channels=self.mid_dim, out_channels=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.squeeze()
        x = torch.sigmoid(x)
        return x


class LocalCNN(nn.Module):

    def __init__(self, m=2):
        super(LocalCNN, self).__init__()
        self.M = m
        self.LANets = nn.ModuleList([LANet(in_dim=512) for _ in range(m)])

    def forward(self, x):
        residu = x
        fs = [module(x) for module in self.LANets]
        x = torch.stack(fs, dim=1)
        x, _ = torch.max(x, dim=1, keepdim=True)
        x = residu * x
        return x


class TransFER(nn.Module):

    def __init__(self, export_layer=4, in_channel=3, width=1):

        super(TransFER, self).__init__()
        layers = [2, 2, 2, 2]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.base = int(64 * width)
        self.export_layer = export_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, self.base, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, self.base * 8, layers[3], stride=2)
        self.LocalCNN = LocalCNN(m=4)
        self.proj = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=384, bias=False)
        self.myclassifier = PreViT384(num_classes=7, dim=384)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, use_ca=False, use_sa=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        att = self.LocalCNN(x)
        att, _ = torch.max(att, dim=1, keepdim=True)
        x = x * att

        x = self.proj(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x = self.myclassifier(x)
        return x, att


def mymodel():
    model = TransFER()
    checkpoint = torch.load('/home/b3432/Code/experiment/zhujunhao/workspace/dacl/resnet18_msceleb.pth',
                            map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


if __name__ == '__main__':
    from torchsummary import summary

    import argparse

    parser = argparse.ArgumentParser(description='PyTorch RAF Training')
    parser.add_argument('--mad_m', type=int, default=3)
    parser.add_argument('--p1', type=float, default=1.0)
    args = parser.parse_known_args()[0]
    model = TransFER(args).cuda()
    summary(model, (3, 224, 224))
