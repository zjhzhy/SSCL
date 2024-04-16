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
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        print('Cannot load image ' + path)


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
        max_samples = 15000

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
        fine_name = imgPath.split('/')[-1]
        imgPath = fine_name.split('.')[0]
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


def pose30_reader(filelist, num_classes):#txt文件
    if num_classes == 7:
        exclude_list = [7, 8, 9, 10]
    else:
        exclude_list = [8, 9, 10]

    f = open(filelist)
    imgList = []
    for line in f.readlines():
        line= line.strip('\n')
        label = line.split('/')[0]
        path =line.split('/')[1]
        path =path.split('.')[0]
        #label, path = line.strip().split(',')
        labels = int(label)
        if labels not in exclude_list:
            imgList.append([path, labels])
    f.close()
    return imgList, None

def pose45_reader(filelist, num_classes):#txt文件
    if num_classes == 7:
        exclude_list = [7, 8, 9, 10]
    else:
        exclude_list = [8, 9, 10]

    f = open(filelist)
    imgList = []
    for line in f.readlines():
        line== line.strip().split(',')
        label = line.split('/')[0]
        path =line.split('/')[1]
        #label, path = line.strip().split(',')
        labels = int(label)
        if labels not in exclude_list:
            imgList.append([path, labels])
    f.close()
    return imgList, None

def occlusion_reader(filelist,num_classes):
    if num_classes == 7:
        exclude_list = [7, 8, 9, 10]
    else:
        exclude_list = [8, 9, 10]
    f = open(filelist)
    imgList = []
    for line in f.readlines():
        line == line.strip().split(',')
        label = line.split('/')[1]
        path = line.split('/')[2]
        # label, path = line.strip().split(',')
        labels = int(label)
        if labels not in exclude_list:
            imgList.append([path, labels])
    f.close()
    return imgList, None



# leibie=['default','pose','occlusion']
class ImageList(data.Dataset):
    def __init__(self, root, fileList, train=True, num_classes=7, list_reader=default_reader,
                 loader=PIL_loader, argument=False, argu_type=0, scale_invariant=False,leibie='train'):
        self.root = root
        self.cls_num = num_classes
        self.imgList, _ = list_reader(fileList, self.cls_num)
        self.loader = loader
        self.fileList = fileList
        self.leibie =leibie
        self.train = train
        self.num_classes = num_classes
        self.num_per_class = self.get_cls_num_list()
        self.samples = self.imgList
        self.flag = argument
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.1))
        ])

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
        expression = switch_expression_zhy(target)  # 标签
        if self.leibie=='training':
            fine_name_jpg = path + '.jpg'
            fine_name_png = path + '.png'
            fine_name_JPG = path + '.JPG'
            fine_name_jpeg = path + '.jpeg'
            fine_name_PNG =path+'.PNG'
            fine_name_JPEG = path + '.JPEG'
            fine_name_tif = path + '.tif'
            fine_name_bmp = path + '.bmp'
            fine_name_Jpeg = path + '.Jpeg'
            path_exist_jpg = os.path.join(self.root, 'training', expression, fine_name_jpg)
            path_exist_JPG = os.path.join(self.root, 'training', expression, fine_name_JPG)
            path_exist_jpeg = os.path.join(self.root, 'training', expression, fine_name_jpeg)
            path_exist_PNG = os.path.join(self.root, 'training', expression, fine_name_PNG)
            path_exist_JPEG = os.path.join(self.root, 'training', expression, fine_name_JPEG)
            path_exist_tif = os.path.join(self.root, 'training', expression, fine_name_tif)
            path_exist_bmp =os.path.join(self.root, 'training', expression, fine_name_bmp)
            path_exist_Jpeg = os.path.join(self.root, 'training', expression, fine_name_Jpeg)
            path_exist_png =os.path.join(self.root, 'training', expression, fine_name_png)
            if os.path.exists(path_exist_jpg):
                path = path_exist_jpg
            elif os.path.exists(path_exist_JPG):
                path = path_exist_JPG
            elif os.path.exists(path_exist_jpeg):
                path = path_exist_jpeg
            elif os.path.exists(path_exist_PNG):
                path =path_exist_PNG
            elif os.path.exists(path_exist_JPEG):
                path =path_exist_JPEG
            elif os.path.exists(path_exist_tif):
                path =path_exist_tif
            elif os.path.exists(path_exist_bmp):
                path =path_exist_bmp
            elif os.path.exists(path_exist_Jpeg):
                path =path_exist_Jpeg
            else:
                path = path_exist_png
        else:
            #expression = switch_expression_zhy(target)  # 标签
            fine_name_jpg = path + '.jpg'
            fine_name_png = path + '.png'
            fine_name_JPG = path + '.JPG'
            fine_name_jpeg = path + '.jpeg'
            fine_name_tif =path +'.tif'
            path_exist_jpg = os.path.join(self.root, 'validation', expression, fine_name_jpg)
            path_exist_JPG = os.path.join(self.root, 'validation', expression, fine_name_JPG)
            path_exist_jpeg = os.path.join(self.root, 'validation', expression, fine_name_jpeg)
            path_exist_png = os.path.join(self.root, 'validation', expression, fine_name_png)
            if os.path.exists(path_exist_jpg):
                path = path_exist_jpg
            elif os.path.exists(path_exist_JPG):
                path = path_exist_JPG
            elif os.path.exists(path_exist_jpeg):
                path = path_exist_jpeg
            elif os.path.exists(path_exist_png):
                path = path_exist_png
            else:
                path = os.path.join(self.root, 'validation', expression, fine_name_tif)

        '''
        if self.leibie=='occlusion':
            fine_name_jpg = path+'.jpg'
            fine_name_png = path+'.png'
            fine_name_JPG = path+'.JPG'
            fine_name_jpeg = path+'.jpeg'
            path_exist_jpg = os.path.join(self.root, 'validation', expression, fine_name_jpg)
            path_exist_JPG = os.path.join(self.root, 'validation', expression, fine_name_JPG)
            path_exist_jpeg = os.path.join(self.root, 'validation', expression, fine_name_jpeg)
            if os.path.exists(path_exist_jpg):
                path = path_exist_jpg
            elif os.path.exists(path_exist_JPG):
                path = path_exist_JPG
            elif os.path.exists(path_exist_jpeg):
                path = path_exist_jpeg
            else:
                path = os.path.join(self.root, 'validation', expression, fine_name_png)
        elif self.leibie=='pose30':
            fine_name_jpg = path + '.jpg'
            fine_name_png = path + '.png'
            fine_name_JPG = path + '.JPG'
            fine_name_jpeg = path + '.jpeg'
            path_exist_jpg = os.path.join(self.root, 'validation', expression, fine_name_jpg)
            path_exist_JPG = os.path.join(self.root, 'validation', expression, fine_name_JPG)
            path_exist_jpeg = os.path.join(self.root, 'validation', expression, fine_name_jpeg)
            if os.path.exists(path_exist_jpg):
                path = path_exist_jpg
            elif os.path.exists(path_exist_JPG):
                path = path_exist_JPG
            elif os.path.exists(path_exist_jpeg):
                path = path_exist_jpeg
            else:
                path = os.path.join(self.root, 'validation', expression, fine_name_png)
        elif self.leibie =='pose45':
            fine_name_jpg = path + '.jpg'
            fine_name_png = path + '.png'
            fine_name_JPG = path + '.JPG'
            fine_name_jpeg = path + '.jpeg'
            path_exist_jpg = os.path.join(self.root, 'validation', expression, fine_name_jpg)
            path_exist_JPG = os.path.join(self.root, 'validation', expression, fine_name_JPG)
            path_exist_jpeg = os.path.join(self.root, 'validation', expression, fine_name_jpeg)
            if os.path.exists(path_exist_jpg):
                path = path_exist_jpg
            elif os.path.exists(path_exist_JPG):
                path = path_exist_JPG
            elif os.path.exists(path_exist_jpeg):
                path = path_exist_jpeg
            else:
                path = os.path.join(self.root, 'validation', expression, fine_name_png)
        '''
        return path

    def get_cls_num_list(self):
        count = [0 for _ in range(self.num_classes)]
        for _, lable in self.imgList:
            count[lable] = count[lable] + 1
        return count

    def __getitem__(self, index):
        imgPath, label = self.imgList[index]
        imgPath=self.transform_path(imgPath,label)
        #imgPath = os.path.join(imgPath, label)
        img = self.loader(imgPath)
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
        return len(self.imgList)


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


def get_dataloader(args):
    print('*******  Running Scale invariant  ********')
    root_path = "/home/lab303/yx/database/affectnet/"
    valid_list = "/home/lab303/yx/database/affectnet/validation.csv"
    train_list = "/home/lab303/yx/database/affectnet/training.csv"
    vaild_path ="/home/lab303/yx/database/affectnet/validation"
    train_path ="/home/lab303/yx/database/affectnet/training"
    pose45_list = "/home/lab303/yx/database/affectnet/pose_45_affectnet_list.txt"
    pose45_data = ImageList(root=root_path, train=False, fileList=pose45_list, num_classes=args.num_classes,
                            list_reader=pose45_reader, argument=True, argu_type=args.argu_type,
                            scale_invariant=args.scale_invariant,leibie='val')
    pose30_list = "/home/lab303/yx/database/affectnet/pose_30_affectnet_list.txt"
    pose30_data = ImageList(root=root_path, train=False, fileList=pose30_list, num_classes=args.num_classes,
                            list_reader=pose30_reader, argument=True, argu_type=args.argu_type,
                            scale_invariant=args.scale_invariant,leibie='val')
    occlusion_list = "/home/lab303/yx/database/affectnet/occlusion_affectnet_list.txt"
    occlusion_data = ImageList(root=root_path, train=False, fileList=occlusion_list, num_classes=args.num_classes,
                               list_reader=occlusion_reader, argument=True, argu_type=args.argu_type,
                               scale_invariant=args.scale_invariant,leibie='val')
    pose30_loader = torch.utils.data.DataLoader(pose30_data, args.batch_size, shuffle=False, num_workers=args.worker,
                                                pin_memory=True)
    pose45_loader = torch.utils.data.DataLoader(pose45_data, args.batch_size, shuffle=False, num_workers=args.worker,
                                                pin_memory=True)
    occlusion_loader = torch.utils.data.DataLoader(occlusion_data, args.batch_size, shuffle=False,
                                                   num_workers=args.worker,
                                                   pin_memory=True)

    val_data = ImageList(root=root_path, train=False, fileList=valid_list, num_classes=args.num_classes,list_reader=default_reader,
                         argument=True, argu_type=args.argu_type, scale_invariant=args.scale_invariant,leibie='val')

    val_loader = torch.utils.data.DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.worker,
                                             pin_memory=True)

    train_dataset = ImageList(root=root_path, train=True, fileList=train_list, num_classes=args.num_classes,list_reader=default_reader,
                              argument=True, argu_type=args.argu_type, scale_invariant=args.scale_invariant,leibie='training')

    cls_num_list = train_dataset.get_cls_num_list()
    print('Train split class wise is :', cls_num_list)

    train_sampler = None
    if args.train_rule == 'Resample':
        train_sampler = ImbalancedDatasetSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.worker, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)

    return train_loader, val_loader, pose30_loader, pose45_loader, occlusion_loader


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
    parser.add_argument("--device", type=str, default='1')
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

    loader = get_dataloader(args)
    for i in loader[3]:
        print(i)
        break
