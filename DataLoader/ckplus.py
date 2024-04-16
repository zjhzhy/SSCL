# -*- coding=utf-8 -*-
import os
from torchvision import transforms
from PIL import Image
import torch
from sklearn.model_selection import StratifiedKFold
import numpy as np


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


class_label = {
    'Neutral': 0,
    'Angry': 6,
    'Contempt': 7,
    'Disgust': 5,
    'Fear': 4,
    'Happy': 1,
    'Sad': 2,
    'Surprise': 3,
}

switcher = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear', 5: 'Disgust', 6: 'Angry',
    7: 'Contempt', 8: 'None', 9: 'Uncertain', 10: 'No-Face'
}


class CKPlus5Flods(object):
    def __init__(self, root, train, num_classes, flod):
        self.root = root
        self.num_classes = num_classes
        self.classes_list = ['Neutral', 'Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
        self.train = train
        self.flod = flod
        self.samples = self.__prepare__()
        self.train_flods = self.samples[1]
        self.val_flods = self.samples[2]
        crop_size = 0.8
        normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        self.transform_strong = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ])

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
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

    def __prepare__(self):
        images = []
        paths = []
        labels = []
        if self.num_classes == 7:  #如果类别为7，删除contempt类别
            self.classes_list.remove('Contempt')
        dirs = os.listdir(self.root)
        for dir in dirs:
            if dir in self.classes_list:
                imgpaths = os.listdir(os.path.join(self.root, dir))#遍历子目录，标签
                for path in imgpaths:
                    imgpath = '%s/%s/%s' % (self.root, dir, path)
                    paths.append(imgpath)
                    labels.append(class_label[dir])
                    images.append((imgpath, class_label[dir]))

        train_list = []
        val_list = []
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        for train_index, test_index in skf.split(paths, labels):
            x = [paths[i] for i in train_index]
            y = [labels[i] for i in train_index]
            val_x = [paths[i] for i in test_index]
            val_y = [labels[i] for i in test_index]
            train_set = []
            val_set = []
            for xx, yy in zip(x, y):
                train_set.append((xx, yy))
            for xx, yy in zip(val_x, val_y):
                val_set.append((xx, yy))
            train_list.append(train_set)
            val_list.append(val_set)
        return images, train_list, val_list

    def __len__(self):
        if self.train:
            return len(self.train_flods[self.flod])
        else:
            return len(self.val_flods[self.flod])

    def __getitem__(self, index):

        if self.train:
            train_set = self.train_flods[self.flod]
            imgPath, label = train_set[index]
            img = pil_loader(imgPath)
            x = self.transform_train(img)
            x_arg = self.transform_strong(img)
            return x, label, x_arg
        else:
            valid_set = self.val_flods[self.flod]
            imgPath, label = valid_set[index]
            img = pil_loader(imgPath)
            x = self.transform_test(img)
            return x, label


def get_five_flods(args, flod):
    data_root = '/home/lab303/zhy/data/CK表情数据集/cohn-kanade/cohn-kanade'
    #data_root = wrap_root(data_root)

    train_set = CKPlus5Flods(data_root, train=True, num_classes=args.num_classes, flod=flod)
    valid_set = CKPlus5Flods(data_root, train=False, num_classes=args.num_classes, flod=flod)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.worker, pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.worker, pin_memory=True)
    return train_loader, val_loader


class CKPlus(object):
    def __init__(self, root, train, num_classes):
        self.root = root
        self.num_classes = num_classes
        self.classes_list = ['Neutral', 'Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
        self.train = train
        self.samples = self.__prepare__()
        crop_size = 0.8
        normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        self.transform_strong = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ])

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
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

    def __prepare__(self):
        images = []
        if self.num_classes == 7:
            self.classes_list.remove('Contempt')
        dirs = os.listdir(self.root)
        for dir in dirs:
            if dir in self.classes_list:
                imgpaths = os.listdir(os.path.join(self.root, dir))
                for path in imgpaths:
                    imgpath = '%s/%s/%s' % (self.root, dir, path)
                    images.append((imgpath, class_label[dir]))
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        imgPath, label = self.samples[index]
        img = pil_loader(imgPath)
        if self.train:
            x = self.transform_train(img)
            x_arg = self.transform_strong(img)
            return x, label, x_arg
        else:
            x = self.transform_test(img)
            return x, label


def get_dataset(args):
    data_root = 'datasets/CK+_Emotion/Train/CK+_Train_crop'
    #data_root = wrap_root(data_root)
    train_set = CKPlus(data_root, train=True, num_classes=args.num_classes)
    valid_set = CKPlus(data_root, train=False, num_classes=args.num_classes)
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
    import argparse

    description = 'arch search emd pair loss '
    # global
    parser = argparse.ArgumentParser(description='PyTorch RAF Training')
    parser.add_argument('--des', type=str, default=description)
    parser.add_argument('--save_freq', type=int, default=-1)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument("--save_best", action='store_true', default=True)
    # parser.add_argument('--suffix', type=str, default='')
    parser.add_argument("--tmp", action='store_true', default=False)
    # parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--device", type=str, default='')
    parser.add_argument("--worker", type=int, default=12)
    # training
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--train_rule', type=str, default='Resample')
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
    parser.add_argument('--pool_size', type=int, default=6)
    parser.add_argument('--target_layers', type=str, default='4', help='emd apply layers, like 3,4')
    parser.add_argument('--temperature', type=float, default=12.5, help='emd dist scale values like alpha')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--scale_invariant', action='store_true', default=False)
    parser.add_argument('--argu_type', type=int, default=0)
    parser.add_argument('--tune', type=str, nargs='*', default='')
    args = parser.parse_known_args()[0]

    _, _ = get_five_flods(args, flod=1)
