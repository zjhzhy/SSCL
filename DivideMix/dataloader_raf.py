from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
import pandas as pd


class raf_dataset(Dataset):

    def __init__(self, root_dir, transform, mode, num_class, pred=[], probability=[], log=''):
        self.root = root_dir
        self.transform = transform
        self.mode = mode
        data_root = '/home/seven/datasets/RAF/'
        if self.mode == 'test':

            self.val_imgs = []
            self.val_labels = {}
            txt_path = os.path.join(data_root, 'basic/EmoLabel/list_patition_label.txt')
            val_data = pd.read_csv(txt_path, header=None, encoding='utf-8', sep=' ')
            val_data = val_data[12271:]

            for (s, label) in zip(val_data[0], val_data[1]):
                s = s.split('.')[0] + '_aligned.jpg'
                path = os.path.join(data_root, 'basic/Image/aligned', s)
                self.val_imgs.append(path)
                self.val_labels[path] = int(label - 1)
        else:
            train_imgs = []
            self.train_labels = {}
            txt_path = os.path.join(data_root, 'basic/EmoLabel/list_patition_label.txt')
            data = pd.read_csv(txt_path, header=None, encoding='utf-8', sep=' ')
            data = data[:12271]
            for (s, label) in zip(data[0], data[1]):
                s = s.split('.')[0] + '_aligned.jpg'
                path = os.path.join(data_root, 'basic/Image/aligned', s)
                train_imgs.append(path)
                self.train_labels[path] = int(label - 1)

            if self.mode == 'all':
                self.train_imgs = train_imgs
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    self.probability = [probability[i] for i in pred_idx]
                    print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))
                    log.write('Numer of labeled samples:%d \n' % (pred.sum()))
                    log.flush()
                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index
        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)


class raf_dataloader():
    def __init__(self, batch_size, num_class, num_workers, root_dir, log):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            all_dataset = raf_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="all",
                                      num_class=self.num_class)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader

        elif mode == 'train':
            labeled_dataset = raf_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="labeled",
                                          num_class=self.num_class, pred=pred, probability=prob, log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)

            unlabeled_dataset = raf_dataset(root_dir=self.root_dir, transform=self.transform_train,
                                            mode="unlabeled", num_class=self.num_class, pred=pred, log=self.log)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = raf_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test',
                                       num_class=self.num_class)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size * 20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = raf_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                       num_class=self.num_class)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size * 20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return eval_loader
