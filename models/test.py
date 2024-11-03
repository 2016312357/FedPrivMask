#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


# celeba
def test_img(net_g, data_loader, args):
    '''
    function: 测试目标模型准确率（模型输出shape为单个概率/logits值,形如(1,)）
    :param net_g: 目标模型
    :param data_loader: 数据集
    :param args: 所有参数
    :return: 准确率
    '''
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.BCELoss()
    # data_loader = DataLoader(datatest, batch_size=args.bs)
    # l = len(data_loader)
    # print(len(data_loader.dataset))
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            # data, target = sample['images'], sample['labels']
            print(idx, data.shape)
            data, target = data.type(torch.FloatTensor), target.type(torch.FloatTensor)
            data, target = data.to(args.device), target.to(args.device)
            # data, target = data.cuda(), target.cuda()
            log_probs = net_g(data)
            test_loss += loss_func(log_probs[:, 0], target)
            # x1, x2, x4, x5, x6, log_probs = net_g(data)
            # test_loss += loss_func(log_probs[:, 0], target)
            log_probs = log_probs.float().cpu().data.numpy()
            target = target.float().cpu().data.numpy()

            for i in range(len(log_probs)):
                if log_probs[i][0] < 0.5:
                    log_probs[i][0] = 0.
                else:
                    log_probs[i][0] = 1.
            # print(log_probs[:,0])
            correct += np.sum(target == log_probs[:, 0])
            # sum up batch loss
            # test_loss += loss_func(log_probs[:, 0], target).item()
            # get the index of the max log-probability

            # correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f})\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def test_mnist(net_g, data_loader, args, prop=0,printing=False):
    '''
    function: 测试目标模型,标签为多分类标签（模型输出shape为概率/logits向量）
    :param net_g: 目标模型
    :param data_loader: 测试集
    :param args: 所有参数
    :return: 测试准确率和损失
    '''
    net_g.eval()
    loss_func = nn.CrossEntropyLoss()
    # lab=[]
    # out=[]

    with torch.no_grad():
        test_loss = 0.
        correct = 0.
        p11, p10 = 0, 0
        target20 = 0
        target21 = 0
        for _, (data, target) in enumerate(data_loader):
            if args.data_name == 'mnist':
                pass
                # if prop == 0:
                #     target = target % 2
                # elif prop == 2:
                #     target[target < 5] = 0
                #     target[target != 0] = 1  # >=5
            elif args.data_name == 'MotionSense' and args.privacy == '':
                target = target[:, prop]
                print(prop, 'Motion sense testing label', target)
            elif isinstance(target, list) and (args.data_name == 'CelebA' or args.data_name == 'lfw') and args.privacy == '':
                # print(target)[tensor([0, 0, 1...]), tensor([0, 1, 0,...])]
                target2 = target[1 - prop]  ####attr2
                target = target[prop]
            else:
                pass
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            test_loss += loss_func(log_probs, target).item()#.long().long()
            log_probs = log_probs.float().cpu().data.numpy()
            target = target.float().cpu().data.numpy()
            correct += np.sum(target == np.argmax(log_probs[:], axis=1))
            # print( np.argmax(log_probs[:], axis=1))
            # print(np.argmax(log_probs[:], axis=1))
            if isinstance(target, list):
                target2 = target2.float().data.numpy()
                all1 = np.ones(len(target))
                all0 = np.zeros(len(target))
                p11 += np.sum(all1[target2 == all1] == (np.argmax(log_probs[:], axis=1)[target2 == all1]))
                p10 += np.sum(all1[target2 == all0] == (np.argmax(log_probs[:], axis=1)[target2 == all0]))
                target21 += np.sum(all1 == target2)
                target20 += np.sum(all0 == target2)

        test_loss /= len(data_loader)#.dataset
        accuracy = correct / len(data_loader.dataset)

    if args.verbose or printing:
        print('Test set: Average loss: {:.4f} \tAccuracy: {}/{} ({:.2f}) \tProp:{}'.format(
            test_loss, correct, len(data_loader.dataset), accuracy, prop))
    # print('Testing MIA auc {} '.format(roc_auc_score(lab, out)))
    if isinstance(target, list):
        unfairness = p11 / (target21 + 1e-6) - p10 / (target20 + 1e-6)  ###fairness metric
        # print("p: ", p11, target21, p10, target20,"demographic parity difference:",target2)
        return accuracy, test_loss, unfairness

    else:
        return accuracy, test_loss,0
