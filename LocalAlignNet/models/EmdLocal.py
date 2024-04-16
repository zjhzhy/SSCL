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


class SelfAttention(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.fc = nn.Linear(inplanes, inplanes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        activation = self.sigmoid(x)
        weight = torch.norm(activation, p=1, dim=1)
        return weight


class Decouple(nn.Module):
    def __init__(self, inplanes, outplanes, M):
        super().__init__()
        self.Flatten = Flatten()
        self.M = M
        self.fc = [nn.Linear(inplanes, outplanes).cuda() for _ in range(M)]
        self.attentions = [SelfAttention(outplanes).cuda() for _ in range(M)]
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.Flatten(x)
        locals = []
        wlist = []
        for i in range(self.M):
            localx = self.fc[i](x)
            w = self.attentions[i](localx)
            wlist.append(w)
            w = w.unsqueeze(1)
            localx = torch.mul(w, localx)
            locals.append(localx)
        wbatch = torch.cat(wlist, dim=0).reshape(len(wlist),-1)
        wbatch = wbatch.T
        # c shape : Batch,M, feature_dim
        c = torch.cat(locals, dim=1).reshape(x.size(0), self.M, -1)
        # c shape : Batch,feature_dim
        c = c.mean(dim=1, keepdim=False)
        return c, wbatch


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


class DeepEMD(nn.Module):
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
        self.Decouple = Decouple(inplanes=512, outplanes=128, M=6)
        self.classifier = nn.Linear(128, args.num_classes)

    def forward(self, batch):
        img = batch[0].to(self.args.device)
        target = batch[1].to(self.args.device)
        feature_list1 = self.encode(img, dense=True)

        emd_logits = []
        if self.training:
            img_aug = batch[2].to(self.args.device)
            feature_list2 = self.encode(img_aug, dense=True)
            for i in self.args.target_layers:
                emd_logits.append(self.emd_forward(feature_list2[eval(i) - 2], feature_list1[eval(i) - 2]))

        x, wbatch = self.Decouple(feature_list1[-1])

        x = self.classifier(x)
        return x, emd_logits, target, wbatch

    def get_weight_vector(self, A, B):
        """
        input :A shape: [batch,channel,H,W]
        return combination -> A weight with repest to B :  [batch,H*W]
        """
        if A.shape.__len__() == 3:
            B = F.adaptive_avg_pool1d(B, 1)
            B = B.repeat(1, 1, A.shape[2])
            combination = (A * B).sum(1)
            combination = combination.view(A.shape[0], -1)
            combination = F.relu(combination) + 1e-3
            return combination
        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])
        combination = (A * B).sum(1)
        combination = combination.view(A.shape[0], -1)
        combination = F.relu(combination) + 1e-3
        return combination

    def emd_forward(self, proto, query):
        """
        计算两个集合的那个emd距离
        proto shape: [batch,channel,H,W]
        """
        proto = proto.squeeze(0)

        weight_1 = self.get_weight_vector(query, proto)
        weight_2 = self.get_weight_vector(proto, query)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)

        similarity_map = self.get_similiarity_map(proto, query)
        if self.args.solver == 'opencv' or (not self.training):
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
        else:
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='qpth')
        return logits

    def get_sfc(self, support):
        """
        依据当前提供的feature，优化原型参数
        """
        support = support.squeeze(0)
        # init the proto
        SFC = support.view(self.args.shot, -1, 640, support.shape[-2], support.shape[-1]).mean(dim=0).clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)

        optimizer = torch.optim.SGD([SFC], lr=self.args.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)

        # crate label for finetune
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label_shot = label_shot.type(torch.cuda.LongTensor)

        with torch.enable_grad():
            for k in range(0, self.args.sfc_update_step):
                rand_id = torch.randperm(self.args.way * self.args.shot).to(self.args.device)
                for j in range(0, self.args.way * self.args.shot, self.args.sfc_bs):
                    selected_id = rand_id[j: min(j + self.args.sfc_bs, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.emd_forward_1shot(SFC, batch_shot.detach())
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
        num_query = similarity_map.shape[0]
        num_node = weight_1.shape[-1]
        num_node2 = weight_2.shape[-1]
        if solver == 'opencv':  # use openCV solver
            for i in range(num_query):
                _, flow = emd_inference_opencv(1 - similarity_map[i, :, :], weight_1[i, :], weight_2[i, :])
                similarity_map[i, :, :] = (similarity_map[i, :, :]) * torch.from_numpy(flow).to(self.args.device)
            temperature = (self.args.temperature / num_node)
            logitis = similarity_map.sum(-1).sum(-1) * temperature
            return logitis

        elif solver == 'qpth':
            weight_1 = weight_1.reshape(num_query, num_node)
            weight_2 = weight_2.reshape(num_query, num_node2)
            _, flows = emd_inference_qpth(1 - similarity_map, weight_1, weight_2, form=self.args.form,
                                          l2_strength=self.args.l2_strength)

            logitis = (flows * similarity_map).view(num_query, flows.shape[-2], flows.shape[-1])
            temperature = (self.args.temperature / num_node)
            logitis = logitis.sum(-1).sum(-1) * temperature
        else:
            raise ValueError('Unknown Solver')

        return logitis

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x

    def get_similiarity_map(self, proto, query):
        """
        proto shape: [batch,channel,H,W]
        return cost: shape:[batch,H*W,H*W]
        """
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)
        proto = proto.permute(0, 2, 1)
        query = query.permute(0, 2, 1)
        feature_size = proto.shape[-2]
        if self.args.metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        elif self.args.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map
        else:
            raise Exception('Not Implementation metric.')
        return similarity_map

    def encode(self, x, dense=True):
        """
        dense:
            True: depend on whether use feature_pyramid
            False: return gap for straight forward
            TODO: unsupport batch of image patches now!
        """
        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            feature_list = self.encoder(x)
            x = feature_list[-1]
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch, x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
            x = x.squeeze(-1)
            return [x]

        else:
            feature_list = self.encoder(x)
            if self.args.downsample:
                feature_list = [downsample(e, self.args.pool_size) for e in
                                feature_list]  # set feature output size same
            if dense is False:
                feature_list = [F.adaptive_avg_pool2d(e, 1) for e in feature_list]
                return feature_list
            if self.args.feature_pyramid is not None:
                feature_list = [self.build_feature_pyramid(e) for e in feature_list]
        return feature_list

    def build_feature_pyramid(self, feature):
        """
        将特征图进行金字塔划分
        feature_pyramid:确定划分的网格大小，如2,3
        out: shape[Batch,channel,1,Node]
        """
        feature_list = []
        for size in self.args.feature_pyramid:
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(feature.shape[0], feature.shape[1], 1, -1))
        out = torch.cat(feature_list, dim=-1)
        return out


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch RAF Training')
    parser.add_argument('--save_freq', type=int, default=-1)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument("--save_best", action='store_true', default=True)
    # parser.add_argument('--suffix', type=str, default='')
    parser.add_argument("--tmp", action='store_true', default=False)
    # parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--worker", type=int, default=12)
    # training
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--num_classes', type=int, default=7)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--m', type=float, default=0.9)
    # about model
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'l2'])
    parser.add_argument('--norm', type=str, default='center', choices=['center'], help='feature normalization')
    parser.add_argument('--deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
    # deepemd fcn only
    parser.add_argument('--feature_pyramid', type=str, default=None, help='you can set it like: 2,3')
    # deepemd sampling only
    parser.add_argument('--num_patch', type=int, default=9)
    # deepemd grid only patch_list
    parser.add_argument('--patch_list', type=str, default='2,3', help='the size of grids at every image-pyramid level')
    parser.add_argument('--patch_ratio', type=float, default=2,
                        help='scale the patch to incorporate context around the patch')
    # slvoer about
    parser.add_argument('--solver', type=str, default='opencv', choices=['opencv', 'qpth'])
    parser.add_argument('--form', type=str, default='L2', choices=['QP', 'L2'])
    parser.add_argument('--l2_strength', type=float, default=0.000001)

    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--downsample', action='store_true', default=True)
    parser.add_argument('--pool_size', type=int, default=7)
    parser.add_argument('--target_layers', type=str, default='', help='emd apply layers, like 3,4')
    parser.add_argument('--temperature', type=float, default=12.5, help='emd dist scale values like alpha')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--scale_invariant', action='store_true', default=False)
    parser.add_argument('--argu_type', type=int, default=0)
    parser.add_argument('--tune', type=str, nargs='*', default='')
    args = parser.parse_known_args()[0]
    model = DeepEMD(args).cuda()
    from torchsummary import summary

    x = torch.randn(size=(64, 3, 244, 244)).cuda()
    arg_x = torch.randn(size=(64, 3, 244, 244)).cuda()
    y = torch.randn(size=(64,)).cuda()

    inputx = [x, y, arg_x]

    model(inputx)
