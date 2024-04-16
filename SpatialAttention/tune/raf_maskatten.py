import sys

sys.path.append('/home/b3432/Code/experiment/zhujunhao/workspace/MoPro')
import os
import warnings
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms

from sklearn.metrics import balanced_accuracy_score

from SpatialAttention.models.MaskAtten import MaskAttNet
import nni


def warn(*args, **kwargs):
    pass


warnings.warn = warn


# drop = 0.5:  0.8742
# drop = 0.2   0.8879
# drop = 0.2   0.8810
# drop = 0.2   0.8823
# drop = 0.2   0.8859

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/home/b3432/Code/experiment/zhujunhao/datasets/RAF/basic',
                        help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--drop', type=float, default=0.0, help='Spation Drop rate.')

    return parser.parse_args()


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None,
                         names=['name', 'label'])

        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:,
                     'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def run_training():
    args = parse_args()
    """nni.variable(nni.quniform(0.0,0.5,0.1), name=args.drop)"""
    args.drop = args.drop
    print('tuning drop_rate:{}'.format(args.drop))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = MaskAttNet(drop=args.drop)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomCrop(224, padding=32)
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms)

    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    val_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)

    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)

            out = model(imgs)

            loss = criterion_cls(out, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (
            epoch, acc, running_loss, optimizer.param_groups[0]['lr']))

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            baccs = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                out = model(imgs)
                loss = criterion_cls(out, targets)

                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)

                baccs.append(balanced_accuracy_score(targets.cpu().numpy(), predicts.cpu().numpy()))
            running_loss = running_loss / iter_cnt
            scheduler.step()

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            """@nni.report_intermediate_result(acc)"""
            best_acc = max(acc, best_acc)
            """@nni.report_final_result(best_acc)"""
            bacc = np.around(np.mean(baccs), 4)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, bacc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))

            if acc > 0.89 and acc == best_acc:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('checkpoints',
                                        "rafdb_epoch" + str(epoch) + "_acc" + str(acc) + "_bacc" + str(bacc) + ".pth"))
                tqdm.write('Model saved.')


if __name__ == "__main__":
    '''@nni.get_next_parameter()'''
    run_training()
