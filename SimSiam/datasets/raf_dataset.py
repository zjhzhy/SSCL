import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os
from PIL import Image
import pandas as pd


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class RAFDataset(Dataset):

    def __init__(self, data_root, train, transform=None, target_transform=None, align=True):
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
        self.train = train
        self.samples = self.__prepare__(data_root, train=train, align=align)
        self.transform = transform
        self.target_transform = target_transform
        self.num_per_class = self.get_cls_num_list()
        self.num_classes = 7

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
            item = (path, label - 1)
            images.append(item)
        return images

    def __getitem__(self, index):
        path, label = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return sample, label

    def __len__(self):
        return len(self.samples)

    def get_cls_num_list(self):
        count = [0 for i in range(7)]
        for _, lable in self.samples:
            count[lable] = count[lable] + 1
        return count


def expand_greyscale(t):
    return t.expand(3, -1, -1)


class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        IMAGE_SIZE = 224
        IMAGE_EXTS = ['.jpg', '.png', '.jpeg']

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
