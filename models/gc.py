'''
Different Methods for Gradient Compression
**********************************************
Input must be a pytorch tensor
**********************************************
'''

import torch
import numpy as np


def quantize(x, input_compress_settings={}):
    '''
    function: 将连续的参数值变为离散值，只保留规定的精度
    :param x:
    :param input_compress_settings:
    :return:
    '''
    compress_settings = {'n': 6}
    compress_settings.update(input_compress_settings)
    # assume that x is a torch tensor
    n = compress_settings['n']
    # print('n:{}'.format(n))
    x = x.float()
    x_norm = torch.norm(x, p=float('inf'))
    sgn_x = ((x > 0).float() - 0.5) * 2
    p = torch.div(torch.abs(x), x_norm)
    renormalize_p = torch.mul(p, n)
    floor_p = torch.floor(renormalize_p)
    compare = torch.rand_like(floor_p)
    final_p = renormalize_p - floor_p
    margin = (compare < final_p).float()
    xi = (floor_p + margin) / n

    Tilde_x = x_norm * sgn_x * xi

    return Tilde_x


def sparse_randomized(x, input_compress_settings={}):
    '''
    function: 随机压缩
    :param x:
    :param input_compress_settings:
    :return:
    '''
    max_iteration = 10000
    compress_settings = {'p': 0.8}
    compress_settings.update(input_compress_settings)
    # p=compress_settings['p']
    # vec_x=x.flatten()
    # out=torch.dropout(vec_x,1-p,train=True)
    # out=out/p
    vec_x = x.flatten()
    d = int(len(vec_x))
    p = compress_settings['p']

    abs_x = torch.abs(vec_x)
    # d=torch.prod(torch.Tensor(x.size()))
    out = torch.min(p * d * abs_x / torch.sum(abs_x), torch.ones_like(abs_x))
    i = 0
    while True:
        i += 1
        # print(i)
        if i >= max_iteration:
            raise ValueError('Too much operations!')
        temp = out.detach()

        cI = 1 - torch.eq(out, 1).float()
        c = (p * d - d + torch.sum(cI)) / torch.sum(out * cI)
        if c <= 1:
            break
        out = torch.min(c * out, torch.ones_like(out))
        if torch.sum(1 - torch.eq(out, temp)):
            break
    z = torch.rand_like(out)
    out = vec_x * (z < out).float() / out
    out = out.reshape(x.shape)
    # out=out.reshape(x.shape)
    return out


def one_bit(x, input_compress_settings={}):
    '''
    function:
    :param x:
    :param input_compress_settings:
    :return:
    '''
    x_norm = torch.norm(x, p=float('inf'))  # max(x1,x2,x3,...)
    sgn_x = ((x > 0).float() - 0.5) * 2  # 1 or -1

    compressed_x = x_norm * sgn_x

    return compressed_x


def sparse_top_k_index(x, input_compress_settings={}):
    '''
    只保留绝对值在top k数目内的参数
    :param x:
    :param input_compress_settings:
    :return:
    '''
    compress_settings = {'k': 0.3}  # 保留30%
    compress_settings.update(input_compress_settings)
    k = compress_settings['k']
    vec_x = x.flatten()
    # d = int(len(vec_x))
    # # print(d)
    k = max(1, int(np.ceil(k)))
    print('sparse top', k)
    indices = torch.abs(vec_x).topk(k, largest=False)[1]  # topk返回的是一个元组，第一个元素指返回的具体值，第二个元素指返回值的索引。
    # out_x = torch.zeros_like(vec_x)
    # out_x[indices] = vec_x[indices]######  top k 保留, 其余置0
    # out_x = out_x.reshape(x.shape)
    return indices


def sparse_top_k(x, input_compress_settings={},dp_noise=None):
    '''
    只保留top k大小的参数
    :param x:
    :param input_compress_settings:
    :return:
    '''
    compress_settings = {'k': 0.3}  # 保留30%
    compress_settings.update(input_compress_settings)
    k = compress_settings['k']
    vec_x = x.flatten()
    if k < 1:###ratio
        d = int(len(vec_x))
        k = int(np.ceil(d * k))
        print('GC pruning, remain the top', k, '/', len(vec_x))
    else:  ### number
        k = int(np.ceil(k))
        print('GC pruning, remain the top number', k, '/', len(vec_x))

    indices = torch.abs(vec_x).topk(k)[1]
    out_x = torch.zeros_like(vec_x)
    out_x[indices] = vec_x[indices]  ######  top k 保留, 其余置0
    if dp_noise is not None:
        noise = torch.normal(0, dp_noise, indices.size()).to(out_x.device)
        print('adding DP', dp_noise, noise.shape)
        out_x[indices] += noise
    out_x = out_x.reshape(x.shape)
    # del indices
    return out_x


def sparse_quantile_k(x, input_compress_settings={},dp_noise=None):
    '''
    只保留top k大小的参数
    :param x:
    :param input_compress_settings:
    :return:
    '''
    compress_settings = {'k': 0.3}  # 保留30%
    compress_settings.update(input_compress_settings)
    k = compress_settings['k']
    k1 = compress_settings['k1']
    vec_x = x.flatten()
    if k <= 1:###ratio
        d = int(len(vec_x))
        k = int(np.ceil(d * k))
        k1 = int(np.ceil(d * k1))
        print('GC pruning, remain the top',k1, k, '/', len(vec_x))
    else:###number
        k = int(np.ceil(k))
        k1 = int(np.ceil(k1))
        # print('remain the top number', k, '/', len(vec_x))
    out_x = torch.zeros_like(vec_x)
    if k<d:##100%
        indices = torch.abs(vec_x).topk(k)[1]
        out_x[indices] = vec_x[indices]  ######  top k 保留, 其余置0
    indices1 = torch.abs(vec_x).topk(k1,largest=False)[1]
    out_x[indices1] = vec_x[indices1]  ######  top k 保留, 其余置0
    # if dp_noise is not None:
    #     noise = torch.normal(0, dp_noise, indices.size()).to(out_x.device)
    #     print('adding DP', dp_noise, noise.shape)
    #     out_x[indices] += noise
    out_x = out_x.reshape(x.shape)
    # del indices, indices1
    return out_x
def binary_top_k(x, input_compress_settings={}):
    '''
    top k大小的参数为1，其余为0
    :param x:
    :param input_compress_settings:
    :return:
    '''
    compress_settings = {'k': 0.3}  # 保留30%
    compress_settings.update(input_compress_settings)
    k = compress_settings['k']
    vec_x = x.flatten()
    d = int(len(vec_x))
    # print(d)
    k = int(np.ceil(d * k))
    # print(d,k)
    indices = torch.abs(vec_x).topk(k)[1]
    out_x = torch.zeros_like(vec_x)
    # print(len(indices)/d)
    out_x[indices] = 1  # vec_x[indices]
    out_x = out_x.reshape(x.shape)
    # print(x.shape)
    return out_x
