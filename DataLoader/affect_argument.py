'''
Aum Sri Sai Ram

Implementation of Affectnet dataset class

Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 10-07-2020
Email: darshangera@sssihl.edu.in


Reference:
1. Mollahosseini, A., Hasani, B. and Mahoor, M.H., 2017. "AffectNet: A database for facial expression, valence,
    and arousal computing in the wild". IEEE Transactions on Affective Computing, 10(1), pp.18-31.

Labels: 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face

No of samples in Manually annoated set for each of the class are below:
0:74874 1:134415 2:25459 3:14090 4:6378 5:3803 6:24882 7:3750

2. For occlusion, pose30 and 45 datasets refer to https://github.com/kaiwang960112/Challenge-condition-FER-dataset based on
Kai Wang, Xiaojiang Peng, Jianfei Yang, Debin Meng, and Yu Qiao , Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
{kai.wang, xj.peng, db.meng, yu.qiao}@siat.ac.cn
"Region Attention Networks for Pose and Occlusion Robust Facial Expression Recognition".
'''

import torch.utils.data as data
from PIL import Image, ImageFile
import os
import torch
from torchvision import transforms
from DataLoader.sampler import ImbalancedDatasetSampler

ImageFile.LOAD_TRUNCATED_IAMGES = True


def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)


def switch_expression(expression_argument):
    switcher = {
        0: 'Neutral',
        1: 'Happiness',
        2: 'Sadness',
        3: 'Surprise',
        4: 'Fear', 5: 'Disgust', 6: 'Anger',
        7: 'Contempt', 8: 'None', 9: 'Uncertain', 10: 'No-Face'
    }
    return switcher.get(expression_argument, 0)

def switch_expression_zhy(expression_argument):
    switcher = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4', 5: '5', 6: '6',
        7: '7', 8: 'None', 9: 'Uncertain', 10: 'No-Face'
    }
    return switcher.get(expression_argument, 0)


def default_reader(fileList, num_classes):
    imgList = []
    if fileList.find('validation.csv') > -1:
        start_index = 1
        max_samples = 100000
    else:
        start_index = 1
        max_samples = 5000

    num_per_cls_dict = dict()
    for i in range(0, num_classes):
        num_per_cls_dict[i] = 0

    if num_classes == 7:
        exclude_list = [7, 8, 9, 10]
    else:
        exclude_list = [8, 9, 10]

    expression_0 = 0
    expression_1 = 0
    expression_2 = 0
    expression_3 = 0
    expression_4 = 0
    expression_5 = 0
    expression_6 = 0
    expression_7 = 0

    '''
    Below Ist two options for occlusion and pose case and 3rd one for general
    '''
    # f = open('../data/Affectnetmetadata/validation.csv', 'r')
    # lines = f.readlines()
    # random.shuffle(lines)

    # training or validation affectnet set

    fp = open(fileList, 'r')
    for line in fp.readlines()[start_index:]:
        imgPath = line.strip().split(',')[0]
        expression = int(line.strip().split(',')[6])

        if expression == 0:
            expression_0 = expression_0 + 1
            if expression_0 > max_samples:
                continue

        if expression == 1:
            expression_1 = expression_1 + 1
            if expression_1 > max_samples:
                continue

        if expression == 2:
            expression_2 = expression_2 + 1
            if expression_2 > max_samples:
                continue

        if expression == 3:
            expression_3 = expression_3 + 1
            if expression_3 > max_samples:
                continue

        if expression == 4:
            expression_4 = expression_4 + 1
            if expression_4 > max_samples:
                continue

        if expression == 5:
            expression_5 = expression_5 + 1
            if expression_5 > max_samples:
                continue

        if expression == 6:
            expression_6 = expression_6 + 1
            if expression_6 > max_samples:
                continue

        if expression == 7:
            expression_7 = expression_7 + 1
            if expression_7 > max_samples:
                continue
                # Adding only list of first 8 expressions
        if expression not in exclude_list:
            imgList.append([imgPath, expression])
            num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1
    fp.close()
    return imgList, num_per_cls_dict


class ImageList(data.Dataset):
    def __init__(self, root, fileList, train=True, num_classes=7, list_reader=default_reader,
                 loader=PIL_loader, argument=False, argu_type=0):
        self.root = root
        self.cls_num = num_classes
        self.imgList, self.num_per_cls_dict = list_reader(fileList, self.cls_num)
        self.loader = loader
        self.fileList = fileList
        self.train = train
        self.num_per_class = self.get_cls_num_list()
        self.num_classes = 7
        self.samples = self.imgList
        self.flag = argument

        crop_size = 0.8
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print('using argu_type:{}'.format(argu_type))
        if argu_type == 0:
            self.transform_strong = self.transform_train
        elif argu_type == 1:
            self.transform_strong = self.transform1
        else:
            raise Exception('Not Implement Argument.')
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def transform_path(self, path, target):
        '''
        expression = switch_expression(target)
        '''
        expression = switch_expression_zhy(target)#标签
        fine_name = path.split('/')[-1]
        if self.train:
            path = os.path.join(self.root, 'training', expression, fine_name)
        else:
            path = os.path.join(self.root, 'validation', expression, fine_name)
        return path

    def get_cls_num_list(self):
        count = [0 for i in range(7)]
        for _, lable in self.imgList:
            count[lable] = count[lable] + 1
        return count

    def __getitem__(self, index):
        imgPath, label = self.imgList[index]
        imgPath = self.transform_path(imgPath, label)
        img = self.loader(imgPath)
        if self.train:
            x = self.transform_train(img)
            x_arg = self.transform_strong(img)
            return x, label, x_arg
        else:
            x = self.transform_test(img)
            return x, label

    def __len__(self):
        return len(self.imgList)


def wrap_root(path):
    d, f = os.path.split(os.path.abspath(__file__))#返回脚本绝对路径
    name = d.split('/')[2]
    b3432 = '/home/b3432/Code/experiment/zhujunhao/'
    seven = '/home/seven/'
    if name == 'b3432':
        return os.path.join(b3432, path)
    elif name == 'seven':
        return os.path.join(seven, path)
    else:
        raise Exception('Not implementation')


def get_dataloader(args):
    '''
    root_path = wrap_root('datasets/AffectNet/224/')
    valid_list = wrap_root('datasets/AffectNet/validation2.csv')
    train_list = wrap_root('datasets/AffectNet/training2.csv')
    '''
    root_path = '/home/lab303/yx/database/affectnet'
    valid_list = '/home/lab303/yx/database/affectnet/validation.csv'
    train_list = '/home/lab303/yx/database/affectnet/zhy_training.csv'
    val_data = ImageList(root=root_path, train=False, fileList=valid_list, num_classes=args.num_classes,
                         argument=True, argu_type=args.argu_type)

    cls_num_list_val =val_data.get_cls_num_list()
    print('val split class wise is :', cls_num_list_val)
    val_loader = torch.utils.data.DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.worker,
                                             pin_memory=True)

    train_dataset = ImageList(root=root_path, train=True, fileList=train_list, num_classes=args.num_classes,
                              argument=True, argu_type=args.argu_type)

    cls_num_list = train_dataset.get_cls_num_list()
    print('Train split class wise is :', cls_num_list)

    train_sampler = None
    if args.train_rule == 'Resample':
        train_sampler = ImbalancedDatasetSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.worker, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)

    return train_loader, val_loader
