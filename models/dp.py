#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import os
import torch
import numpy as np
import random
import torch.nn.functional as F

"""
How to apply alg1:
alg1(model, l2_lambda=0.001, delta=1e-5, epsilon=0.1)
"""


def clipping(clipthr, w):
    # clipping L2 norm of the parameters within a threshold
    if get_1_norm(w) > clipthr:
        w_local = copy.deepcopy(w)
        for i in w.keys():
            if 'weight' in i or 'bias' in i:
                w_local[i] = w_local[i] * clipthr / get_1_norm(w)
    else:  # no clip
        w_local = copy.deepcopy(w)
    return w_local  

# get the L2-norm of the parameters
def get_1_norm(params_a):
    sum = 0
    if isinstance(params_a, np.ndarray):
        sum += pow(np.linalg.norm(params_a, ord=2), 2)
    else:
        for i in params_a.keys():
            if 'weight' in i or 'bias' in i:
                # print('not clip', i)
                if len(params_a[i]) == 1:
                    sum += pow(np.linalg.norm(params_a[i].cpu().numpy(), ord=2), 2)
                else:
                    a = copy.deepcopy(params_a[i].cpu().numpy())
                    for j in a:
                        x = copy.deepcopy(j.flatten())
                        sum += pow(np.linalg.norm(x, ord=2), 2)
    norm = np.sqrt(sum)
    return norm

def alg1(net, l2_lambda, delta, epsilon, device=None, dp_mechanism='gauss'):
    '''
    net: the model needed to add DP noise
    l2_lambda: L2 sensitivity of the model
    delta, epsilon: differential privacy parameters
    device: specify the cuda or cpu device
    dp_mechanism: what kind of noise: default gaussian
    
    reutrn: the differential private model
    '''

    # net_global = copy.deepcopy(net)
    w = net.state_dict()

    # calculate the sigma of DP noise
    if dp_mechanism == 'gauss':
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * l2_lambda/ epsilon
        print('adding gaussian noise', sigma, epsilon)
    else:
        print('Such noise has not been implemented yet')
        sigma = None
        
    
    
    # add dp noise for each layer
    if sigma is not None:
        w = clipping(l2_lambda, w)  # clip the L2 sensitivity of the model
        for k in w.keys():
            if 'weight' in k or 'bias' in k:
                print('adding noise to', k)
                if dp_mechanism == 'gauss':
                    noise = np.random.normal(0, sigma, w[k].size())
                    noise = torch.from_numpy(noise).float()
                    if device is not None:
                        noise=noise.to(device)
                w[k] += noise
        net.load_state_dict(w)  # load the noised model  
    
    return net