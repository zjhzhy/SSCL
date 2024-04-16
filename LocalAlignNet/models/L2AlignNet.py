import torch
import torch.nn as nn
import torch.nn.functional as F
from LocalAlignNet.models.emd_utils import *
from LocalAlignNet.models.resnet import *
import math


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return x


def initialistion_sub_module(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def downsample(x, output_size):
    """
    TODO: convolution downsample need！
    """
    x = F.adaptive_avg_pool2d(x, output_size=output_size)
    return x


class L2Align(nn.Module):
    """
    batch compair
    """

    def __init__(self, args, mode='train'):
        super().__init__()
        self.mode = mode
        self.args = args
        if args.backbone == 'resnet18':
            self.encoder = resnet18(pretrained=args.pretrained, encoder=True)
        if args.backbone == 'resnet34':
            self.encoder = resnet34(pretrained=args.pretrained, encoder=True)
        if args.backbone == 'resnet50':
            self.encoder = resnet50(pretrained=args.pretrained, encoder=True)
        if args.backbone == 'resnet101':
            self.encoder = resnet101(pretrained=args.pretrained, encoder=True)
        if self.encoder is None:
            raise Exception('No backbone')
        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(self.encoder.base * 8 * self.encoder.block.expansion, args.num_classes)
        )

    def forward(self, batch):
        img = batch[0].to(self.args.device)
        target = batch[1].to(self.args.device)
        feature_list1 = self.encoder(img)
        l2_loss = None
        if self.training:
            img_aug = batch[2].to(self.args.device)
            feature_list2 = self.encoder(img_aug)
            l2_loss = self.l2_forward(feature_list1[-1], feature_list2[-1])
        logits1 = self.classifier(feature_list1[-1])
        return logits1, l2_loss, target

    def l2_forward(self, proto, query):

        proto = proto.reshape(proto.size(0), -1)
        query = query.reshape(query.size(0), -1)
        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)
        l2loss = nn.functional.mse_loss(proto, query, reduce='mean')
        return l2loss

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x
