# -*- coding=utf-8 -*-

import numpy as np


def Batchnorm_simple_for_train(x, gamma, beta, bn_param):
    running_mean = bn_param['running_mean']
    running_var = bn_param['running_var']
    m = bn_param['momentun']
    result = 0.

    x_mean = x.mean(axis=0)
    x_var = x.var(axis=0)

    running_mean = m * running_mean + (1 - m) * x_mean
    running_var = m * running_var + (1 - m) * x_var

    x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
    result = gamma * x_normalized + beta
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return result, bn_param

def Batchnorm_simple_for_test(x, gamma, beta, bn_param):
    running_mean = bn_param['running_mean']  #shape = [B]
    running_var = bn_param['running_var']    #shape = [B]

    x_normalized=(x-running_mean )/np.sqrt(running_var +eps)       # 归一化
    results = gamma * x_normalized + beta            # 缩放平移

    return results , bn_param