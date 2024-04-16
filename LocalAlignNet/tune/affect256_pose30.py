# _*_ coding: utf-8 _*_
"""
"""
import sys

sys.path.append('/home/b3432/Code/experiment/zhujunhao/workspace/MoPro')
sys.path.append('/home/seven/workspace/MoPro')
sys.path.append('/content/drive/My Drive/colab/MoPro')
import torch.nn as nn
from torch.optim import lr_scheduler
import torch
import os
from LocalAlignNet.models.EMDAlignNet import DeepEMD
import argparse
import utils
import pandas as pd
from tools.meter.average_meter import AverageMeter
from tools.meter.time_meter import TimeMeter
from tools.metric import accuracy
import time
from prefetch_generator import BackgroundGenerator
from DataLoader.affect256_scale import get_dataloader
import json
from config import Config
import logging
import sys
import nni


def main():
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
    parser.add_argument('--batch_size', type=int, default=256)
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
    parser.add_argument('--pool_size', type=int, default=7)
    parser.add_argument('--target_layers', type=str, default='4', help='emd apply layers, like 3,4')
    parser.add_argument('--temperature', type=float, default=12.5, help='emd dist scale values like alpha')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--scale_invariant', action='store_true', default=False)
    parser.add_argument('--argu_type', type=int, default=0)
    parser.add_argument('--tune', type=str, nargs='*', default='')
    args = parser.parse_known_args()[0]

    logsuffix = ''
    # tune trail space
    if 'scale' in args.tune:
        logger.info('tuning pretrained.')
        print('tuning scale_variant.')
        """nni.variable(nni.choice('true', 'false'), name=is_scale)"""
        is_scale = ''
        if is_scale == 'true':
            args.scale_invariant = True
        elif is_scale == 'false':
            args.scale_invariant = False
        logsuffix = logsuffix + 'scale_invariant_{}'.format(is_scale)

    if 'pretrained' in args.tune:
        logger.info('tuning pretrained.')
        print('tuning pretrained.')
        """nni.variable(nni.choice('true', 'false'), name=is_pretrained)"""
        is_pretrained = ''
        if is_pretrained == 'true':
            args.pretrained = True
        elif is_pretrained == 'false':
            args.pretrained = False

    if 'backbone' in args.tune:
        logger.info('tuning backbone')
        print('tuning backbone:{}'.format(args.backbone))
        """nni.variable(nni.choice('resnet18', 'resnet34','resnet50'), name=args.backbone)"""
        args.backbone = args.backbone

    if 'pool_size' in args.tune:
        logger.info('tuning pool_size')
        print('tuning pool_size:{}'.format(args.pool_size))
        """nni.variable(nni.choice(4,5,6,7), name=args.pool_size)"""
        args.pool_size = int(args.pool_size)
        logsuffix = logsuffix + 'poolsize{}'.format(args.pool_size)

    if 'target_layers' in args.tune:
        print('tuning target_layers:{}'.format(args.target_layers))
        """nni.variable(nni.choice('3', '2'), name=args.target_layers)"""
        args.target_layers = args.target_layers
        logsuffix = logsuffix + 'targetlayers{}'.format(args.target_layers)

    if 'alpha' in args.tune:
        print('tuning alpha:{}'.format(args.alpha))
        """nni.variable(nni.quniform(0.0,1.0,0.2), name=args.alpha)"""
        args.alpha = args.alpha
        logsuffix = logsuffix + 'alpha{:.2f}'.format(args.alpha)
    # if 'lr' in args.tune:
    #     print('tuning lr:{}'.format(args.lr))
    #     """@nni.variable(nni.choice('0.1', '0.01'), name=args.lr)"""
    #     args.lr = args.lr
    #     args.lr = float(args.lr)
    # mkdir and set version
    logsuffix += '_nonundersample' + str(nni.get_sequence_id())
    cfg = Config.cfg
    file_name = utils.get_cur_file_name(__file__)
    cfg = utils.set_version(cfg, version=file_name, suffix=logsuffix, tmp=args.tmp, args=args)
    _print = cfg.logger.print_log
    _print(json.dumps(vars(args), separators=('\n', ':')))
    _print('nni experiments id:{}'.format(str(nni.get_experiment_id())))
    _print('nni sequence id:{}'.format(str(nni.get_sequence_id())))
    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    cfg.device = args.device

    best_acc = 0.0
    best_acc_epoch = 0
    # load models
    model = DeepEMD(args).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.m,
                                weight_decay=args.wd)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)

    # load data
    train_loader, val_loader, pose30_loader, pose45_loader, occlusion_loader = get_dataloader(args)
    val_loader = pose30_loader
    first_batch_gap_time = time.time()
    for epoch in range(0, 0 + args.max_epoch):

        check_batch = -1
        _print('**' * 50)

        model.train()
        epoch_time = TimeMeter()
        data_time = AverageMeter('6.3f')
        losses = AverageMeter(':.4f')
        top1 = AverageMeter(':6.2f')
        end = time.time()
        d_t = time.time()

        pbar = enumerate(BackgroundGenerator(train_loader))
        for i, batch in pbar:
            if i == 0:
                first_batch_gap_time = time.time() - first_batch_gap_time
                _print('first_batch_gap_time:{}'.format(first_batch_gap_time))
            data_time.update(time.time() - end)
            d_t = time.time() - d_t
            s_t = time.time()

            # forward
            logit, emd_dists, target = model(batch)

            # loss computation
            cls_loss = criterion(logit, target)
            emd_loss = 0.0
            for dist in emd_dists:
                dist = dist.mean(0)
                emd_loss += dist
            loss = cls_loss - emd_loss * args.alpha

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(logit, target, topk=(1,))
            losses.update(loss.item(), target.size(0))
            top1.update(acc[0].item(), target.size(0))

            s_t = time.time() - s_t
            if check_batch != -1 and i % check_batch == 0:
                print(
                    'Train Epoch:{:03d} [{}/{} ({:.0f}%)]  --Loss:{:.4f} --acc1: {:6.2f} --data_time_second: {} --update_time:{}' \
                        .format(epoch,
                                i * len(target),
                                len(train_loader.dataset),
                                100. * i / len(train_loader),
                                loss.item(),
                                acc[0].item(),
                                d_t,
                                s_t
                                ))
            end = time.time()
            d_t = time.time()
        epoch_time.stop()
        _print(
            'Train Epoch:{:03d} --Train Acc: {:6.2f} --Loss: {:.4f} --Time: {}'.format(epoch, top1.avg, losses.avg,
                                                                                       epoch_time))

        scheduler.step()
        # ==========================================record=========================================
        # tensor board record loss and acc
        if args.save_freq != -1 and epoch % args.save_freq == 0:
            model_path = '{}_{:03d}.models'.format(cfg.version, epoch)
            model_path = os.path.join(cfg.path.ckpt, model_path)
            torch.save({
                'models': model.state_dict(),
            }, model_path)

        # eval
        if epoch % args.eval_freq == 0:

            model.eval()
            epoch_time = TimeMeter()
            losses = AverageMeter(':.4f')
            top1 = AverageMeter(':6.2f')
            with torch.no_grad():
                # pbar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader))
                pbar = enumerate(BackgroundGenerator(val_loader))

                for i, batch in pbar:
                    logit, _, target = model(batch)
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(logit, target).item()
                    acc = accuracy(logit, target, topk=(1,))
                    losses.update(loss, target.size(0))
                    top1.update(acc[0].item(), target.size(0))
            epoch_time.stop()
            _print(
                'Test: --Accuracy:{:6.2f} --Average Loss: {:.4f} --Time: {}'.format(top1.avg, losses.avg, epoch_time))
            valid_loss, valid_acc = losses.avg, top1.avg
            """@nni.report_intermediate_result(valid_acc)"""

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_acc_epoch = epoch
                if args.save_best and epoch > 0:
                    model_path = os.path.join(cfg.path.ckpt, 'best')
                    torch.save({
                        'models': model.state_dict(),
                    }, model_path)
            _print('>>>>>best_acc:{:.3f}------best_acc_epoch:{:03d}>>>>>'.format(best_acc, best_acc_epoch))
            first_batch_gap_time = time.time()
    cfg.logger.rename(metric=best_acc)
    """@nni.report_final_result(best_acc)"""
    return best_acc


if __name__ == '__main__':
    '''@nni.get_next_parameter()'''
    logger = logging.getLogger('DEMSearch_AutoML_affect256')
    try:
        main()
    except Exception as exception:
        logger.exception(exception)
        raise
