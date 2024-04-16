import torch
import torch.nn as nn
import torch.nn.functional as F
from LocalAlignNet.models.emd_utils import *
from LocalAlignNet.models.resnet import *
import math


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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = torch.nn.Sequential(
            conv3x3(3, 16, 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Dropout2d(p=0.1),
        )

        self.conv2 = torch.nn.Sequential(
            conv3x3(16, 32, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout2d(p=0.1),
        )
        self.conv3 = torch.nn.Sequential(
            conv3x3(32, 64, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout2d(p=0.1),
        )
        self.conv4 = torch.nn.Sequential(
            conv3x3(64, 80, 2),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(0.01),
            nn.Dropout2d(p=0.1),
        )

        self.conv5 = torch.nn.Sequential(
            conv3x3(80, 128, 2),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = torch.nn.Sequential(
            nn.ConvTranspose2d(128, 80, kernel_size=3, stride=2,
                               padding=1, bias=False, output_padding=1),
            DoubleConv(80, 80)
        )
        self.conv2 = torch.nn.Sequential(
            nn.ConvTranspose2d(80, 64, kernel_size=3, stride=2,
                               padding=1, bias=False, output_padding=1),
            DoubleConv(64, 64))
        self.conv3 = torch.nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, bias=False, output_padding=1),
            DoubleConv(32, 32))
        self.conv4 = torch.nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, bias=False, output_padding=1),
            DoubleConv(16, 16))
        self.conv5 = torch.nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2,
                               padding=1, bias=False, output_padding=1),
            DoubleConv(3, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x1 = self.encoder.conv1(x)
        x2 = self.encoder.conv1(x1)
        x3 = self.encoder.conv1(x2)
        x4 = self.encoder.conv1(x3)
        x5 = self.encoder.conv1(x4)

        xx5 = self.encoder.conv1(x5)
        xx4 = self.encoder.conv1(xx5)
        xx3 = self.encoder.conv1(xx4)
        xx2 = self.encoder.conv1(xx3)
        xx1 = self.encoder.conv1(xx2)


        return xx1


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





if __name__ == '__main__':
    from torchsummary import summary

    m = AE().cuda()
    summary(m, input_size=(3, 224, 224), batch_size=1)
