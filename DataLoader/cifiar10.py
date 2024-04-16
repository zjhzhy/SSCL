# -*- coding=utf-8 -*-
import torch
import torchvision
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
rain_data = torchvision.datasets.CIFAR10(root='C:/Users/Administrator/.keras/datasets', train=True,
                                        download=False, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='C:/Users/Administrator/.keras/datasets', train=False,
                                       download=False, transform=transform)
## 将训练集重新划分为验证集和训练集
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size*num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sample = SubsetRandomSampler(train_idx)
valid_sample = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,
                                          sampler=train_sample,num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,
                                          sampler=valid_sample,num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,num_workers=num_workers)
