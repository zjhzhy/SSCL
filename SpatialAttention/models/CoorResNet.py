import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from SpatialAttention.attention.CoordAttention import CoordAtt


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class CoorBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(CoorBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.atten = CoordAtt(inplanes, planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.atten(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SeResNet(nn.Module):

    def __init__(self, args, layers, export_layer=4, in_channel=3, width=1, num_class=1000):

        super(SeResNet, self).__init__()
        self.args = args
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)
        self.export_layer = export_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(CoorBasicBlock, self.base, layers[0], stride=1)
        self.layer2 = self._make_layer(CoorBasicBlock, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(CoorBasicBlock, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(CoorBasicBlock, self.base * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(self.base * 8 * CoorBasicBlock.expansion, num_class)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, reduction=self.args.reduction))
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

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


def coor_resnet18(args, pretrained='ms1m'):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SeResNet(args, [2, 2, 2, 2], num_class=args.num_classes)
    if pretrained == 'ms1m':
        model.load_state_dict(torch.load('/home/b3432/Code/experiment/zhujunhao/downloads/ms1m_res18.pkl'),
                              strict=False)
    elif pretrained == 'resnet18_msceleb':
        model.load_state_dict(torch.load('/home/b3432/Code/experiment/zhujunhao/workspace/dacl/resnet18_msceleb.pth'),
                              strict=False)
    elif pretrained == 'imagenet':
        state_dict = torch.load('/home/b3432/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth')
        model.load_state_dict(state_dict, strict=False)
    return model
