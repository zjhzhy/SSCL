# -*- coding=utf-8 -*-

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from DataLoader.sampler import ImbalancedDatasetSampler


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class SFEW(Dataset):

    def __init__(self, mode, train, data_root='/home/seven/datasets/SFEW', num_classes=7):
        """
        :param data_root:
        :param train:
        :param transform:
        :param target_transform:
        :param align:
        """
        self.mode = mode
        self.train = train
        self.num_classes = num_classes
        print('num_classes is {}'.format(self.num_classes))
        print('mode is {}'.format(self.mode))
        if self.num_classes == 7:
            self.class_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Angry', 'Neutral']
            self.name2label = {
                'Surprise': 0,
                'Fear': 1,
                'Disgust': 2,
                'Happiness': 3,
                'Sadness': 4,
                'Angry': 5,
                'Neutral': 6,
            }
        else:
            raise Exception('no specific num_classes.')

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

    def get_cls_num_list(self):
        count = [0 for i in range(self.num_classes)]
        for _, lable in self.samples:
            count[lable] = count[lable] + 1
        return count

    def __prepare__(self, data_root):
        if not os.path.isdir(data_root):
            raise Exception
        if self.mode == 'train':
            path = os.path.join(data_root, './Train/Train_Aligned_Faces')
        elif self.mode == 'valid':
            path = os.path.join(data_root, './Val/Val_Aligned_Faces')
        # elif self.mode == 'test':
        #     path = os.path.join(data_root, './Test/imgs')
        # elif self.mode == 'test_all':
        #     path1 = os.path.join(data_root, './Val/imgs')
        #     path2 = os.path.join(data_root, './Test/imgs')
        #     images = self.make_datasets(path1, self.name2label)
        #     images.extend(self.make_datasets(path2, self.name2label))
        #     # print(len(images))
        #     return images
        else:
            raise Exception('no specific data_root or train mode.')
        images = self.make_datasets(path, self.name2label)
        return images

    def make_datasets(self, path, class_to_index):
        images = []
        dir = os.path.expanduser(path)
        for target in sorted(os.listdir(dir)):
            # print(target)
            if target in self.class_names:
                d = os.path.join(dir, target)
                if not os.path.isdir(d):
                    continue
                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        item = (path, class_to_index[target])
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


def get_dataset(args=None):
    data_root = '/home/b3432/Code/experiment/zhujunhao/datasets/SFEW'
    data_root = wrap_root(data_root)
    train_set = SFEW(mode='train', train=True, data_root=data_root, num_classes=args.num_classes)
    val_set = SFEW(mode='valid', train=False, data_root=data_root, num_classes=args.num_classes)
    return train_set, val_set


def get_dataloader(args=None):
    train_set, valid_set = get_dataset(args=args)
    train_sampler = None
    if hasattr(args, 'train_rule'):
        if args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=4, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)
    return train_loader, val_loader


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
    # valid_set = get_dataset(mode='test', args=args)
    # val_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=True,
    #                                          num_workers=4,
    #                                          pin_memory=True)
    train_loader, val_loader = get_dataloader(args)
    # print(valid_set.samples)
    for i, result in enumerate(val_loader):
        img = result['img']
        label = result['label']
