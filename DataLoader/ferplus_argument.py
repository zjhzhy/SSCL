# -*- coding=utf-8 -*-

import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from DataLoader.sampler import ImbalancedDatasetSampler


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class FERPlus(Dataset):

    def __init__(self, data_root='/home/lab303/yx/database/FerPlus', mode='train',
                 align=False, num_classes=7):
        """
        :param data_root:
        :param train:
        :param transform:
        :param target_transform:
        :param align:
        """

        self.align = align
        self.mode = mode
        if self.mode == 'train':
            self.train = True
        else:
            self.train = False
        self.num_classes = num_classes
        print('num_classes is {}'.format(self.num_classes))
        print('mode is {}'.format(self.mode))
        if self.num_classes == 7:
            self.class_names = ['0','1','2','3','4','5','6']
            self.name2label = {#4是開心
                 'Surprise': 0,
                'Fear': 1,
                'Disgust': 2,
                'Happy': 3,
                'Sadness': 4,
                'Anger': 5,
                'Neutral': 6,
            }
            self.name_test_label={
                '6': 'Surprise',
                '3': 'Fear',
                '2': 'Disgust',
                '4': 'Happy',
                '5': 'Sadness',
                '0': 'Anger',
                '7': 'Neutral',
            }
        elif self.num_classes == 8:
            self.class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sadness', 'Anger', 'Neutral', 'Contempt']
            self.name2label = {
                'Surprise': 0,
                'Fear': 1,
                'Disgust': 2,
                'Happy': 3,
                'Sadness': 4,
                'Anger': 5,
                'Neutral': 6,
                'Contempt': 7,
            }
        else:
            raise Exception('no specific num_classes.')

        self.switcher = [3, 4, 5, 1, 2, 6, 0]
        #self.switcher_test = [6, 3, 2, 4, 5, 0, 7]
        crop_size = 0.8
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_strong = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.samples = self.__prepare__(data_root)

    def get_cls_num_list(self):#计数
        count = [0 for i in range(self.num_classes)]
        for _, lable in self.samples:
            count[lable] = count[lable] + 1
        return count

    def __prepare__(self, data_root):
        if not os.path.isdir(data_root):
            raise Exception
        if self.mode == 'train':
            path = os.path.join(data_root, 'Training')
        elif self.mode == 'valid':
            path = os.path.join(data_root, 'PublicTest')
        elif self.mode == 'test':
            path = os.path.join(data_root, 'PrivateTest')
        elif self.mode == 'test_all':
            path1 = os.path.join(data_root, 'PrivateTest')
            path2 = os.path.join(data_root, 'PublicTest')
            images = self.make_datasets(path1, self.name2label)
            images.extend(self.make_datasets(path2, self.name2label))
            return images
        else:
            raise Exception('no specific data_root or train mode.')
        images = self.make_datasets(path, self.class_names)
        return images

    def make_datasets(self, path, class_to_index):
        images = []
        for target in (os.listdir(path)):
            # print(target)
            if target in self.class_names:
                d = os.path.join(path, target)
                if not os.path.isdir(d):
                    continue
                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        #item = (path, class_to_index[target])
                        item = (path, target)
                        images.append(item)
        return images

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = pil_loader(path)
        if self.train:
            x = self.transform_train(img)
            x_arg = self.transform_strong(img)
            return x, label, x_arg
        else:
            x = self.transform_test(img)
            return x, label

    def __len__(self):
        return len(self.samples)


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


def get_dataloader(args=None):
    #root_path = wrap_root('datasets/ferplus')
    root_path='/home/lab303/zhy/data/FerPlus'
    train_set = FERPlus(data_root=root_path, mode='train', num_classes=args.num_classes)
    valid_set = FERPlus(data_root=root_path, mode='valid', num_classes=args.num_classes)
    test_set = FERPlus(data_root=root_path, mode='test', num_classes=args.num_classes)
    train_sampler = None
    if hasattr(args, 'train_rule'):
        if args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.worker, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.worker,
                                             pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.worker,
                                              pin_memory=True)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pytorch1.5', add_help=False)
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument("--tmp", action='store_true')
    parser.add_argument("--device", type=str, default='cuda:0')
    # training
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=48)
    # parser.add_argument('--lr_step', type=list, default=[40, 60])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--num_classes', type=int, default=7)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--m', type=float, default=0.9)
    # colab asset dir
    parser.add_argument("--colab", action='store_true')
    parser.add_argument("--colab_data_root", type=str, default='/content/raf/')
    args = parser.parse_known_args()[0]
    get_dataloader(args)
