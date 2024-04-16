from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models
import random
from itertools import repeat


class SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """

    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])  # 默认沿着中间所有的shape

        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


class DANNet(nn.Module):
    def __init__(self, num_class=7, num_head=4):
        super(DANNet, self).__init__()

        resnet = models.resnet18(True)

        checkpoint = torch.load('/home/b3432/Code/experiment/zhujunhao/workspace/dacl/resnet18_msceleb.pth')
        resnet.load_state_dict(checkpoint['state_dict'], strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head

        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        # self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)
        att = self.att(x)
        out = x * att
        # out = self.bn(out)
        out = self.fc(out)
        return out


class MaskAttenAlignNet(nn.Module):
    def __init__(self, num_class=7, drop=0.8):
        super(MaskAttenAlignNet, self).__init__()

        resnet = models.resnet18(True)

        checkpoint = torch.load('/home/b3432/Code/experiment/zhujunhao/workspace/dacl/resnet18_msceleb.pth')
        resnet.load_state_dict(checkpoint['state_dict'], strict=True)
        self.drop = drop
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.ca = ChannelAttention(512)
        self.sa = SpatialAttention()
        self.mask_sa = MasklSpatialAttention(ratio=self.drop)
        self.fc = nn.Linear(512 * 2, num_class)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)
        x = x * self.ca(x)
        att, mask_att = self.mask_sa(x)

        out1 = x * att
        # out = self.bn(out)
        out1 = self.avg_pool(out1)
        out1 = out1.reshape(out1.shape[0], -1)

        out2 = x * mask_att
        # out = self.bn(out)
        out2 = self.avg_pool(out2)
        out2 = out2.reshape(out2.shape[0], -1)

        out = self.fc(torch.cat([out1, out2], -1))

        return out1, out2, out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
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


class MasklSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, ratio=1.0):
        super(MasklSpatialAttention, self).__init__()
        self.r = ratio
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)

        maxvalue = self.pool(x.detach())
        mask = maxvalue * self.r
        mask = mask.repeat(1, 1, x.shape[2], x.shape[3])
        mask = torch.where(x > mask, 0, 1)
        x2 = x * mask

        return x, x2

    # class CrossAttentionHead(nn.Module):


#     def __init__(self):
#         super().__init__()
#         self.sa = SpatialAttention()
#         self.drop = SpatialDropout(0.5)
#         self.ca = ChannelAttention()
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         sa = self.sa(x)
#         self.drop(sa, noise_shape=(x.shape[0], x.shape[1], 1, 1))
#         ca = self.ca(sa)
#
#         return ca
#
#
# class SpatialAttention(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.conv1x1 = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=1),
#             nn.BatchNorm2d(256),
#         )
#         self.conv_3x3 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#         )
#         self.conv_1x3 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
#             nn.BatchNorm2d(512),
#         )
#         self.conv_3x1 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
#             nn.BatchNorm2d(512),
#         )
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         y = self.conv1x1(x)
#         y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
#         y = y.sum(dim=1, keepdim=True)
#         out = x * y
#
#         return out
#
#
# class ChannelAttention(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.attention = nn.Sequential(
#             nn.Linear(512, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(inplace=True),
#             nn.Linear(32, 512),
#             nn.Sigmoid()
#         )
#
#     def forward(self, sa):
#         sa = self.gap(sa)
#         sa = sa.view(sa.size(0), -1)
#         y = self.attention(sa)
#         out = sa * y
#
#         return out


if __name__ == '__main__':
    from torchsummary import summary

    model = MaskAttenAlignNet().cuda()
    summary(model, input_size=(3, 224, 224))
