# -*- coding=utf-8 -*-

import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def wrap_root(path):
    d, f = os.path.split(os.path.abspath(__file__))
    name = d.split('/')[2]
    b3432 = '/home/b3432/Code/experiment/zhujunhao/'
    seven = '/home/seven/'
    if name == 'b3432':
        return os.path.join(b3432, path)
    elif name == 'seven':
        return os.path.join(seven, path)
    else:
        raise Exception('Not implementation')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class RAFDataset(Dataset):

    def __init__(self, data_root, train, align=True, crop_size=0.8, argu_type=0, scale_invariant=False):
        """
        :param data_root:
        :param train:
        :param transform:
        :param target_transform:
        :param align:
        """

        self.changelist = []
        self.class_dict = {
            1: 'Surprise',
            2: 'Fear',
            3: 'Disgust',
            4: 'Happiness',
            5: 'Sadness',
            6: 'Anger',
            7: 'Neutral',
        }
        self.switcher = [3, 4, 5, 1, 2, 6, 0]
        self.train = train
        self.samples = self.__prepare__(data_root, train=train, align=align)
        self.num_per_class = self.get_cls_num_list()
        self.num_classes = 7
        self.scale_invariant = scale_invariant
        self.argu_type = argu_type
        crop_size = 0.8
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])

        self.other_transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        ])

        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __prepare__(self, data_root, train, align):
        if not os.path.isdir(data_root):
            raise Exception
        images = []
        txt_path = os.path.join(data_root, 'basic/EmoLabel/list_patition_label.txt')
        data = pd.read_csv(txt_path, header=None, encoding='utf-8', sep=' ')
        if train:
            data = data[0:12271]
        else:
            data = data[12271:]
        for (s, label) in zip(data[0], data[1]):
            if align:
                s = s.split('.')[0] + '_aligned.jpg'
                path = os.path.join(data_root, 'basic/Image/aligned', s)
            else:
                path = os.path.join(data_root, 'basic/Image/original', s)
            label = self.switcher[label - 1]
            item = (path, label)
            images.append(item)
        return images

    def __getitem__(self, index):
        imgPath, label = self.samples[index]
        img = pil_loader(imgPath)
        if self.train:
            x = self.base_transform(img)
            if self.scale_invariant:
                # print('scale_invariant')
                x_arg = self.other_transform(x)
            else:
                # print('scale_variant')
                x_arg = self.base_transform(img)
                x_arg = self.other_transform(x_arg)
            x = self.post_transform(x)
            x_arg = self.post_transform(x_arg)
            return x, label, x_arg
        else:
            x = self.transform_test(img)
            return x, label

    def __len__(self):
        return len(self.samples)

    def get_cls_num_list(self):
        count = [0 for i in range(7)]
        for _, lable in self.samples:
            count[lable] = count[lable] + 1
        return count


def get_distribution(data_root):
    txt_path = os.path.join(data_root, 'basic/EmoLabel/list_patition_label.txt')
    data = pd.read_csv(txt_path, header=None, encoding='utf-8', sep=' ')
    all_count = {}
    for i in data[1]:
        if i in all_count:
            all_count[i] = all_count[i] + 1
        else:
            all_count[i] = 1
    print(all_count)
    return all_count


def get_dataset(args):
    data_root = 'datasets/RAF'
    data_root = wrap_root(data_root)
    train_set = RAFDataset(data_root, train=True, argu_type=args.argu_type, scale_invariant=args.scale_invariant)
    valid_set = RAFDataset(data_root, train=False, argu_type=args.argu_type, scale_invariant=args.scale_invariant)
    return train_set, valid_set


def get_dataloader(args=None):
    train_set, valid_set = get_dataset(args)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.worker, pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.worker, pin_memory=True)
    return train_loader, val_loader


if __name__ == '__main__':
    get_distribution(data_root='/home/seven/datasets/RAF')
