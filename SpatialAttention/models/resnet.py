import torch
import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from SpatialAttention.models.TransFormer import ViT, PreViT384

__all__ = ['resnet18_cbam4', 'resnet18_cbam34', 'resvit', 'resprevit384']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, export_layer=4, in_channel=3, width=1, num_class=1000):
        self.inplanes = 64
        self.block = block
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)
        self.export_layer = export_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.classifier = nn.Linear(self.base * 8 * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feature = None
        x = self.layer1(x)
        x = self.layer2(x)
        if self.export_layer == 2:
            feature = x
        x = self.layer3(x)
        if self.export_layer == 3:
            feature = x
        x = self.layer4(x)
        if self.export_layer == 4:
            feature = x
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x, feature


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_ca=True, use_sa=True):
        super(CBAMBlock, self).__init__()
        self.use_ca = use_ca
        self.use_sa = use_sa

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_ca:
            out = self.ca(out) * out
        if self.use_sa:
            out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CBAMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAMBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class ResPreVit384(nn.Module):

    def __init__(self, layers, export_layer=4, in_channel=3, width=1, num_class=1000, use_ca=True, use_sa=True):

        super(ResPreVit384, self).__init__()
        self.inplanes = 64
        self.use_ca = use_ca
        self.use_sa = use_sa
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)
        self.export_layer = export_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(CBAMBlock, self.base, layers[0], stride=1)
        self.layer2 = self._make_layer(CBAMBlock, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(CBAMBlock, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(CBAMBlock, self.base * 8, layers[3], stride=2, use_ca=use_ca, use_sa=use_sa)
        self.proj = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=384, bias=False)
        self.classifier = PreViT384(num_classes=7, dim=384)

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
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, use_ca=use_ca, use_sa=use_sa))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_ca=use_ca, use_sa=use_sa))

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
        x = self.proj(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x = self.classifier(x)
        return x


class ResVit(nn.Module):

    def __init__(self, layers, export_layer=4, in_channel=3, width=1, num_class=1000, use_ca=True, use_sa=True):

        super(ResVit, self).__init__()
        self.inplanes = 64
        self.use_ca = use_ca
        self.use_sa = use_sa
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)
        self.export_layer = export_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(CBAMBlock, self.base, layers[0], stride=1)
        self.layer2 = self._make_layer(CBAMBlock, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(CBAMBlock, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(CBAMBlock, self.base * 8, layers[3], stride=2, use_ca=use_ca, use_sa=use_sa)

        self.classifier = ViT(num_classes=7, dim=512)

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
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, use_ca=use_ca, use_sa=use_sa))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_ca=use_ca, use_sa=use_sa))

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
        x = x.reshape(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x = self.classifier(x)
        return x


class SA4_ResNet18(nn.Module):

    def __init__(self, layers, export_layer=4, in_channel=3, width=1, num_class=1000, use_ca=True, use_sa=True):

        super(SA4_ResNet18, self).__init__()
        self.inplanes = 64
        self.use_ca = use_ca
        self.use_sa = use_sa
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)
        self.export_layer = export_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(CBAMBlock, self.base, layers[0], stride=1)
        self.layer2 = self._make_layer(CBAMBlock, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(CBAMBlock, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(CBAMBlock, self.base * 8, layers[3], stride=2, use_ca=use_ca, use_sa=use_sa)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(self.base * 8 * CBAMBlock.expansion, num_class)

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
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, use_ca=use_ca, use_sa=use_sa))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_ca=use_ca, use_sa=use_sa))

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

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


class SA34_ResNet18(nn.Module):

    def __init__(self, layers, export_layer=4, in_channel=3, width=1, num_class=1000, use_ca=True, use_sa=True):

        super(SA34_ResNet18, self).__init__()
        self.inplanes = 64
        self.use_ca = use_ca
        self.use_sa = use_sa
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)
        self.export_layer = export_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(CBAMBlock, self.base, layers[0], stride=1)
        self.layer2 = self._make_layer(CBAMBlock, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(CBAMBlock, self.base * 4, layers[2], stride=2, use_ca=use_ca, use_sa=use_sa)
        self.layer4 = self._make_layer(CBAMBlock, self.base * 8, layers[3], stride=2, use_ca=use_ca, use_sa=use_sa)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(self.base * 8 * CBAMBlock.expansion, num_class)

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
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, use_ca=use_ca, use_sa=use_sa))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_ca=use_ca, use_sa=use_sa))

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

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


def resnet18_msceleb():
    print('loading resnet18 msceleb model.')
    msceleb_model = torch.load('/home/b3432/Code/experiment/zhujunhao/workspace/dacl/resnet18_msceleb.pth')
    state_dict = msceleb_model['state_dict']
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(state_dict, strict=False)
    return model


def resprevit384(args):
    model = ResPreVit384([2, 2, 2, 2], use_ca=args.use_ca, use_sa=args.use_sa)
    model.load_state_dict(torch.load('/home/b3432/Code/experiment/zhujunhao/downloads/ms1m_res18.pkl'), strict=False)
    return model


def resvit(args):
    print('loading resnet18 msceleb model by dongliang .')
    model = ResVit([2, 2, 2, 2], use_ca=args.use_ca, use_sa=args.use_sa)
    model.load_state_dict(torch.load('/home/b3432/Code/experiment/zhujunhao/downloads/ms1m_res18.pkl'), strict=False)
    return model


def resnet18_cbam4(args):
    print('loading resnet18 msceleb model by dongliang .')
    model = SA4_ResNet18([2, 2, 2, 2], use_ca=args.use_ca, use_sa=args.use_sa)
    model.load_state_dict(torch.load('/home/b3432/Code/experiment/zhujunhao/downloads/ms1m_res18.pkl'), strict=False)
    return model


def resnet18_cbam34(args):
    print('loading resnet18 msceleb model by dongliang .')
    model = SA34_ResNet18([2, 2, 2, 2], use_ca=args.use_ca, use_sa=args.use_sa)
    model.load_state_dict(torch.load('/home/b3432/Code/experiment/zhujunhao/downloads/ms1m_res18.pkl'), strict=False)
    return model


#
# def resnet18_msceleb():
#     print('loading resnet18 msceleb model.')
#     msceleb_model = torch.load('/home/b3432/Code/experiment/zhujunhao/workspace/dacl/resnet18_msceleb.pth')
#     state_dict = msceleb_model['state_dict']
#     model = ResNetEncoder(BasicBlock, [2, 2, 2, 2])
#     model.load_state_dict(state_dict, strict=False)
#     return model
#


if __name__ == '__main__':
    from torchsummary import summary

    import argparse

    parser = argparse.ArgumentParser(description='PyTorch RAF Training')
    parser.add_argument('--use_ca', action='store_true', default=False)
    parser.add_argument('--use_sa', action='store_true', default=False)
    args = parser.parse_known_args()[0]
    puremodel = ResPreVit384([2, 2, 2, 2], use_ca=args.use_ca, use_sa=args.use_sa)

    model = resprevit384(args)

    print(model.layer1._modules['0'].conv1.weight.data[0][0][0])
    print(puremodel.layer1._modules['0 '].conv1.weight.data[0][0][0])
