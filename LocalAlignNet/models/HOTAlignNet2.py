import torch
import torch.nn as nn
import torch.nn.functional as F
from LocalAlignNet.models.emd_utils import *
from LocalAlignNet.models.resnet import *
from LocalAlignNet.models.LightFace.EfficientFace import efficient_face_encoder
from LocalAlignNet.models.mobilenet import mobilenet_v2
from LocalAlignNet.models.vgg import vgg16
import math
import numpy as np


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
        if args.backbone == 'resnet18_msceleb':
            self.encoder = resnet18_msceleb()
        if args.backbone == 'ms1m_cbam_res18':
            self.encoder = ms1m_cbam_res18()
        if args.backbone == 'ms1m_res18':
            self.encoder = ms1m_res18()
        if args.backbone == 'efficient_face':
            self.encoder = efficient_face_encoder()
        if args.backbone == 'mobilenetv2':
            self.encoder = mobilenet_v2(pretrained=args.pretrained)
        if args.backbone == 'vgg16':
            self.encoder = vgg16(pretrained=args.pretrained)

        if self.encoder is None:
            raise Exception('No backbone')

        if args.backbone == 'efficient_face':
            self.classifier = nn.Sequential(
                Flatten(),
                nn.Linear(1024, args.num_classes)
            )
        elif args.backbone == 'mobilenetv2':
            self.classifier = nn.Sequential(
                nn.Linear(1280, args.num_classes)
            )
        elif args.backbone == 'vgg16':
            self.classifier = nn.Sequential(
                Flatten(),
                nn.Linear(512, args.num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                Flatten(),
                nn.Linear(self.encoder.base * 8 * self.encoder.block.expansion, args.num_classes)
            )

    def forward(self, batch):
        img = batch[0].to(self.args.device)
        target = batch[1].to(self.args.device)
        feature_list1 = self.encode(img, dense=True)[-1]
        hot_dist = 0
        inner_loss = 0
        if self.training:
            img_aug = batch[2].to(self.args.device)
            feature_list2 = self.encode(img_aug, dense=True)[-1]
            cost_map = self.emd_forward(feature_list2, feature_list1)
            hot_dist = self.Hot(cost_map, target)
            inner_loss = self.inner_class_emd(cost_map)
        logits1 = self.classifier(feature_list1)
        return logits1, hot_dist, target, inner_loss

    def inner_class_emd(self, emd_logits):
        inner_logits = torch.diag(emd_logits, 0)
        inner_loss = - inner_logits.mean()
        return inner_loss

    def Hot(self, emd_logits, target):
        # emd_logits : N,N
        # target : N ,  1 -- >  N , classes
        t = F.one_hot(target, num_classes=self.args.num_classes) > 0.5
        N, C = t.shape
        index_tensor = torch.arange(0, N)
        cost_matrix = emd_logits.detach().cpu().numpy()
        all_dist = 0
        for k in range(C):
            mask = t[:, k].squeeze()
            pos_idx = torch.masked_select(index_tensor, mask).to(self.args.device)
            neg_idx = torch.masked_select(index_tensor, ~mask).to(self.args.device)
            PN = pos_idx.shape[0]
            NN = neg_idx.shape[0]
            if PN > 1 and NN > 1:
                src_weight = np.ones(shape=(PN, 1), dtype=np.float32)
                dst_weight = np.ones(shape=(NN, 1), dtype=np.float32)
                cost_map = np.zeros(shape=(PN, NN), dtype=np.float32)
                for i in range(PN):
                    for j in range(NN):
                        cost_map[i, j] = cost_matrix[pos_idx[i], neg_idx[j]]

                src_weight = src_weight * (src_weight.shape[0] / src_weight.sum())
                dst_weight = dst_weight * (dst_weight.shape[0] / dst_weight.sum())
                _, _, _flow = cv2.EMD(src_weight, dst_weight, cv2.DIST_USER, cost_map)

                _flow = torch.from_numpy(_flow).to(self.args.device)
                _flow = _flow.view(_flow.shape[0] * _flow.shape[1])
                flow = torch.zeros_like(emd_logits, dtype=torch.float32, requires_grad=False).to(self.args.device)
                pos_idx_ = pos_idx.repeat_interleave(NN)
                neg_idx_ = neg_idx.repeat(PN)
                flow[pos_idx_, neg_idx_] = _flow
                dist = emd_logits * flow / (PN * NN + 1e-5)
                all_dist += dist.sum()
        return all_dist

    def get_weight_vector(self, A, B):

        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)

        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
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
        num_proto = similarity_map.shape[1]
        num_node = weight_1.shape[-1]
        if solver == 'opencv':  # use openCV solver

            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
                    similarity_map[i, j, :, :] = (1 - similarity_map[i, j, :, :]) * torch.from_numpy(flow).to(
                        self.args.device)
            temperature = (self.args.temperature / num_node)
            logitis = similarity_map.sum(-1).sum(-1) * temperature
            return logitis

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x

    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        if self.args.metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        elif self.args.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map
        else:
            raise Exception('Not implementaion args.metric')
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



