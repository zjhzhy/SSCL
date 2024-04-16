# -*- coding=utf-8 -*-
"""
experiment results:
train_emdv2.y:
    settings:
    {
    "des":"emd  pair loss "
    "max_epoch":100
    "batch_size":64
    "pretrained":true
    "num_classes":7
    "lr":0.01
    "wd":0.0005
    "m":0.9
    "metric":"cosine"
    "norm":"center"
    "deepemd":"fcn"
    "feature_pyramid":null
    "num_patch":9
    "patch_list":"2,3"
    "patch_ratio":2
    "solver":"opencv"
    "form":"L2"
    "l2_strength":1e-06
    "temperature":12.5
    "alpha":0.4 tunning
    }
    alpha:
        1.0  - 87.353
        0.8  - 87.321
        0.6  - 87.777
        0.4  - 87.256
        0.2  - 87.419
        0.0  - 86.408
"""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch RAF Training')
    parser.add_argument('--tune', type=str, nargs='*', default='')
    args = parser.parse_known_args()[0]
    print(vars(args))
