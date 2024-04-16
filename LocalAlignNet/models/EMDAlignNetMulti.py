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
        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(self.encoder.base * 8 * self.encoder.block.expansion, args.num_classes)
        )

    def forward(self, batch):
        img = batch[0].to(self.args.device)
        target = batch[1].to(self.args.device)
        feature_list1 = self.encode(img, dense=True)

        emd_logits2 = []
        emd_logits3 = []
        emd_logits4 = []
        if self.training:
            img_aug2 = batch[2].to(self.args.device)
            feature_list2 = self.encode(img_aug2, dense=True)
            for i in self.args.target_layers:
                emd_logits2.append(self.emd_forward(feature_list2[eval(i) - 2], feature_list1[eval(i) - 2]))

            img_aug3 = batch[3].to(self.args.device)
            feature_list3 = self.encode(img_aug3, dense=True)
            for i in self.args.target_layers:
                emd_logits3.append(self.emd_forward(feature_list3[eval(i) - 2], feature_list1[eval(i) - 2]))

            img_aug4 = batch[4].to(self.args.device)
            feature_list4 = self.encode(img_aug4, dense=True)
            for i in self.args.target_layers:
                emd_logits4.append(self.emd_forward(feature_list4[eval(i) - 2], feature_list1[eval(i) - 2]))

        logits1 = self.classifier(feature_list1[-1])
        return logits1, emd_logits2, target, emd_logits3, emd_logits4

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
