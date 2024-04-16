# -*- coding=utf-8 -*-
import os

# os.system(
#     'python /home/b3432/Code/experiment/zhujunhao/workspace/MoPro/LocalAlignNet/train/train_emdv2.py \
#     --alpha=0.4 --gpu=1 --suffix=alpha0.4')
#
# os.system(
#     'python /home/b3432/Code/experiment/zhujunhao/workspace/MoPro/LocalAlignNet/train/train_emdv2.py \
#     --alpha=0.2 --gpu=1 --suffix=alpha0.2')
#
# os.system(
#     'python /home/b3432/Code/experiment/zhujunhao/workspace/MoPro/LocalAlignNet/train/train_emdv2.py \
#     --alpha=0.0 --gpu=1 --suffix=alpha0.0')

os.system('python /home/b3432/Code/experiment/zhujunhao/workspace/MoPro/LocalAlignNet/train/tune_EMDAlignNet.py \
    --tune target_layers --gpu 1 --suffix layers ')

# os.system(
#     'python /home/b3432/Code/experiment/zhujunhao/workspace/MoPro/LocalAlignNet/train/train_EMDAlignNet.py \
#     --alpha 0.6 --target_layers 4 --gpu 1 --suffix=layer4_alpha0.6')
