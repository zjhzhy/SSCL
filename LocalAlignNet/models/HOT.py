import cv2
import torch
import torch.nn.functional as F
from qpth.qp import QPFunction, QPSolvers

import torch.nn as nn
from torch import linalg as LA


def Hott():
    emd_logits = torch.randint(low=1, high=10, size=(16, 16)).cuda()
    target = torch.arange(0, 16) % 7
    t = F.one_hot(target, num_classes=7) > 0.5
    N, C = t.shape
    index_tensor = torch.arange(0, 16)
    cost_matrix = emd_logits.detach().cpu().numpy()
    all_dist = 0
    for k in range(C):
        mask = t[:, k].squeeze()
        pos_idx = torch.masked_select(index_tensor, mask).cuda()
        neg_idx = torch.masked_select(index_tensor, ~mask).cuda()
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

            _flow = torch.from_numpy(_flow).cuda()
            _flow = _flow.view(_flow.shape[0] * _flow.shape[1])
            flow = torch.zeros_like(emd_logits, dtype=torch.float32, requires_grad=False).cuda()
            pos_idx_ = pos_idx.repeat_interleave(NN)
            neg_idx_ = neg_idx.repeat(PN)
            flow[pos_idx_, neg_idx_] = _flow
            dist = emd_logits * flow / (PN * NN + 1e-5)
            all_dist += dist.sum()
    return all_dist


def normalize_feature(x):
    x = x - x.mean(-1).unsqueeze(-1)
    return x


def HOT(weight1, weight2):
    # weight1 : [ # NUM_SET , # NUM_Node, # Feature_Dim ]

    """"
    params:
    weight1 : 一个样本，共三个维度，【NS,NN,FD】表示【NS个子集，每个子集有NN个节点，每个节点有FD维特征】
    weight2 : 同weight1
    need param:
        src_dist : [ # NUM_SET ]  [ N ]
        dst_dist : [ # NUM_SET ]  [ M ]
        cost_map : [ # NUM_SET, # NUM_SET ]  [N , M]
    """
    # 特征去中心化
    weight1 = normalize_feature(weight1)
    weight2 = normalize_feature(weight2)

    N = weight1.shape[0]
    M = weight2.shape[0]

    # 最外层的权重设置为均匀分布
    src_weight = torch.ones(size=(N,)).cuda()
    dst_weight = torch.ones(size=(M,)).cuda()
    # 最外层的代价函数由内层的EMD距离计算所得（即OT()）
    cost_map = torch.zeros(size=(N, M)).cuda()
    for i in range(N):
        for j in range(M):
            cost_map[i, j] = OT(weight1[i, :, :], weight2[j, :, :])
    _, flow = emd_inference_opencv(cost_map, src_weight, dst_weight)
    dist = cost_map * torch.from_numpy(flow).cuda()
    # TODO: 这里最后的距离采用的是求和的形式，显然最外层节点设置的数目会影响结果大小，是否有必要去除这种偏置。
    dist = dist.sum()
    return dist


def cross_reference_weight(A, B):
    """
    input :A shape:
    return combination -> A weight with respect to B :  [M, N, H*W] ==> [M, N, #Node]
    """
    M = A.shape[0]
    B = B.mean(dim=0, keepdim=True)
    B = B.repeat(M, 1)
    combination = (A * B).sum(dim=1)
    combination = combination.view(M)
    combination = F.relu(combination) + 1e-3
    return combination


def get_cos_map(src, dst):
    """
    src: [ N, #Node]
    dst: [ M, #Node]
    return :
        cost: [ N, M ]
    """

    # N , M
    dot_m = torch.matmul(src, dst.T)
    src_norm = LA.norm(src, dim=1, keepdim=True).repeat(1, src.shape[1])
    dst_norm = LA.norm(dst, dim=1, keepdim=True).repeat(1, dst.shape[1])
    norm_mat = torch.matmul(src_norm, dst_norm.T) + 1e-8
    cosine_mat = dot_m / norm_mat
    cost_map = 1 - cosine_mat
    # cosine_mat : [ N, M ]
    return cost_map


def OT(weight1, weight2):
    """
    内层计算方法：
    1. 节点权重采用DeepEMD的cross-referencce机制
    2. 代价采用cosine距离度量
    3. opencv计算flow后，计算总的传输距离，因为代价的定义由输入特征生成，所以优化过程中可以对权重产生梯度。
    """
    src_weight = cross_reference_weight(weight1, weight2)
    dst_weight = cross_reference_weight(weight2, weight1)
    cost_map = get_cos_map(weight1, weight2)
    _, flow = emd_inference_opencv(cost_map, src_weight, dst_weight)
    emd = cost_map * torch.from_numpy(flow).cuda()
    emd = emd.mean(-1).mean(-1)
    return emd


def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()
    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow


def emd_inference_opencv_test(distance_matrix, weight1, weight2):
    distance_list = []
    flow_list = []

    for i in range(distance_matrix.shape[0]):
        cost, flow = emd_inference_opencv(distance_matrix[i], weight1[i], weight2[i])
        distance_list.append(cost)
        flow_list.append(torch.from_numpy(flow))

    emd_distance = torch.Tensor(distance_list).cuda().double()
    flow = torch.stack(flow_list, dim=0).cuda().double()

    return emd_distance, flow


if __name__ == '__main__':
    random_seed = True
    if random_seed:
        pass
    else:

        seed = 1
        import random
        import numpy as np

        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    batch_size = 1
    num_node = 3
    form = 'QP'  # in [ 'L2', 'QP' ]
    weight1 = torch.rand(10, 8, 6).cuda()
    weight2 = torch.rand(10, 8, 6).cuda()
    dist = HOT(weight1, weight2)
    print(dist)
    # cosine_distance_matrix = torch.rand(batch_size, num_node, num_node).cuda()
    #
    # weight1 = torch.rand(batch_size, num_node).cuda()
    # weight2 = torch.rand(batch_size, num_node).cuda()
    #
    # emd_distance_cv, cv_flow = emd_inference_opencv_test(cosine_distance_matrix, weight1, weight2)
    # # emd_distance_qpth, qpth_flow = emd_inference_qpth(cosine_distance_matrix, weight1, weight2, form=form)
    #
    # emd_score_cv = ((1 - cosine_distance_matrix) * cv_flow).sum(-1).sum(-1)
    # emd_score_qpth = ((1 - cosine_distance_matrix) * qpth_flow).sum(-1).sum(-1)
    # print('emd difference:', (emd_score_cv - emd_score_qpth).abs().max())

    pass
