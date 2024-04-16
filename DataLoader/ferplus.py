# -*- coding=utf-8 -*-

import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from DataLoader.sampler import ImbalancedDatasetSampler
import csv
from DataLoader.ferplus_util import Rect
import sys


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class FERPlus(Dataset):

    def __init__(self, data_root='/home/seven/datasets/ferplus', mode='train',
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
        self.training_mode = 'majority'
        if self.mode == 'train':
            self.train = True
        else:
            self.train = False
        self.num_classes = num_classes
        print('num_classes is {}'.format(self.num_classes))
        print('mode is {}'.format(self.mode))
        if self.num_classes == 7:
            self.class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sadness', 'Anger', 'Neutral']
            self.name2label = {
                'Surprise': 0,
                'Fear': 1,
                'Disgust': 2,
                'Happy': 3,
                'Sadness': 4,
                'Anger': 5,
                'Neutral': 6,
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
            path = os.path.join(data_root, './Training')
        elif self.mode == 'valid':
            path = os.path.join(data_root, './PublicTest')
        elif self.mode == 'test':
            path = os.path.join(data_root, './PrivateTest')
        elif self.mode == 'test_all':
            path1 = os.path.join(data_root, './PrivateTest')
            path2 = os.path.join(data_root, './PublicTest')
            images = self.make_datasets(path1, self.name2label)
            images.extend(self.make_datasets(path2, self.name2label))
            return images
        else:
            raise Exception('no specific data_root or train mode.')
        images = self.make_datasets(path, self.name2label)
        return images

    def load_folders(self, mode):
        '''
        Load the actual images from disk. While loading, we normalize the input data.
        '''
        self.data = []
        for folder_name in self.sub_folders:
            folder_path = os.path.join(self.base_folder, folder_name)
            in_label_path = os.path.join(folder_path, self.label_file_name)
            with open(in_label_path) as csvfile:
                emotion_label = csv.reader(csvfile)
                for row in emotion_label:
                    # load the image
                    image_path = os.path.join(folder_path, row[0])
                    image_data = Image.open(image_path)
                    image_data.load()

                    # face rectangle
                    box = list(map(int, row[1][1:-1].split(',')))
                    face_rc = Rect(box)

                    emotion_raw = list(map(float, row[2:len(row)]))
                    emotion = self._process_data(emotion_raw, mode)
                    idx = np.argmax(emotion)
                    if idx < self.emotion_count:  # not unknown or non-face
                        emotion = emotion[:-2]
                        emotion = [float(i) / sum(emotion) for i in emotion]
                        self.data.append((image_path, image_data, emotion, face_rc))

    def _process_target(self, target):
        '''
        Based on https://arxiv.org/abs/1608.01041 the target depend on the training mode.

        Majority or crossentropy: return the probability distribution generated by "_process_data"
        Probability: pick one emotion based on the probability distribtuion.
        Multi-target:
        '''
        if self.training_mode == 'majority' or self.training_mode == 'crossentropy':
            return target
        elif self.training_mode == 'probability':
            idx = np.random.choice(len(target), p=target)
            new_target = np.zeros_like(target)
            new_target[idx] = 1.0
            return new_target
        elif self.training_mode == 'multi_target':
            new_target = np.array(target)
            new_target[new_target > 0] = 1.0
            epsilon = 0.001  # add small epsilon in order to avoid ill-conditioned computation
            return (1 - epsilon) * new_target + epsilon * np.ones_like(target)

    def _process_data(self, emotion_raw, mode):
        '''
        Based on https://arxiv.org/abs/1608.01041, we process the data differently depend on the training mode:

        Majority: return the emotion that has the majority vote, or unknown if the count is too little.
        Probability or Crossentropty: convert the count into probability distribution.abs
        Multi-target: treat all emotion with 30% or more votes as equal.
        '''
        size = len(emotion_raw)
        emotion_unknown = [0.0] * size
        emotion_unknown[-2] = 1.0

        # remove emotions with a single vote (outlier removal)
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size

        if mode == 'majority':
            # find the peak value of the emo_raw list
            maxval = max(emotion_raw)
            if maxval > 0.5 * sum_list:
                emotion[np.argmax(emotion_raw)] = maxval
            else:
                emotion = emotion_unknown  # force setting as unknown
        elif (mode == 'probability') or (mode == 'crossentropy'):
            sum_part = 0
            count = 0
            valid_emotion = True
            while sum_part < 0.75 * sum_list and count < 3 and valid_emotion:
                maxval = max(emotion_raw)
                for i in range(size):
                    if emotion_raw[i] == maxval:
                        emotion[i] = maxval
                        emotion_raw[i] = 0
                        sum_part += emotion[i]
                        count += 1
                        if i >= 8:  # unknown or non-face share same number of max votes
                            valid_emotion = False
                            if sum(emotion) > maxval:  # there have been other emotions ahead of unknown or non-face
                                emotion[i] = 0
                                count -= 1
                            break
            if sum(
                    emotion) <= 0.5 * sum_list or count > 3:  # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
                emotion = emotion_unknown  # force setting as unknown
        elif mode == 'multi_target':
            threshold = 0.3
            for i in range(size):
                if emotion_raw[i] >= threshold * sum_list:
                    emotion[i] = emotion_raw[i]
            if sum(emotion) <= 0.5 * sum_list:  # less than 50% of the votes are integrated, we discard this example
                emotion = emotion_unknown  # set as unknown

        return [float(i) / sum(emotion) for i in emotion]

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


def get_dataloader(args=None):
    root_path = wrap_root('datasets/ferplus')
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
