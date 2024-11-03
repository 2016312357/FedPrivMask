#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated learning arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=15, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size: B")
    # parser.add_argument('--batch_size', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")
    # parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    # parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    # parser.add_argument('--n_covered', type=int, default=2, help="classes covered by each device")
    parser.add_argument('--data_name', type=str, default='CelebA', help="cifar100-MIA dataset name")
    parser.add_argument('--optim', type=str, default='fedavg', help="fedprox,fedavg")
    parser.add_argument('--mu', type=float, default=0.01, help="mu for fedprox")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--arch', type=str, default='vgg16', help='model architecture')

    # # lotteryFL pruning arguments
    # parser.add_argument('--prune_mode', type=bool, default=False,
    #                     help='pruning or not')
    # parser.add_argument('--prune_percent', type=float, default=10,
    #                     help='pruning percent')
    # parser.add_argument('--prune_start_acc', type=float, default=0.8,
    #                     help='pruning start acc')
    # parser.add_argument('--prune_end_rate', type=float, default=0.5,
    #                     help='pruning end rate')
    # parser.add_argument('--mask_ratio', type=float, default=0.5,
    #                     help='mask ratio')

    ####Non-IID
    parser.add_argument('--n_class', type=int, default=1,
                        help="number of image classes per client have")
    parser.add_argument('--num_samples', type=int, default=10,
                        help="non-iid, number of images per class per client have")
    parser.add_argument('--unbalance_rate', type=float, default=1.0,
                        help="class unbalance rate within each client's training data")
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                            non-i.i.d setting (use 0 for equal splits)')

    # Fedmask Optimization options.

    parser.add_argument('--lr_decay_every', type=int,
                        help='Step decay every this many epochs')
    parser.add_argument('--save_prefix', type=str, default='./checkpoints/',
                        help='Location to save model')
    parser.add_argument('--no_mask', default=False,
                        help='Used for running baselines, does not use any masking')
    parser.add_argument('--defense', type=str, default='none')
    # ###Learning rate
    parser.add_argument('--lr_mask', type=float, default=0.00001,
                        help='Learning rate for mask')
    parser.add_argument('--lr_mask_decay_every', type=int, default=15,
                        help='Step decay every this many epochs')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Mask threshold')


    parser.add_argument('--lr_classifier', type=float, default=0.0005,
                        help='Learning rate for classifier')
    parser.add_argument('--lr_classifier_decay_every', type=int, default=15,
                        help='Step decay every this many epochs')

    parser.add_argument('--lr_decay_factor', type=float, default=0.1,
                        help='Multiply lr by this much every step of decay')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay')
    parser.add_argument('--train_bn', default=False, 
                        help='train batch norm or not')

    parser.add_argument('--selected', default=False,
                        help='select mask or not')
    parser.add_argument('--tau', type=float, default=1,
                        help='temperature for selected')







    # Masking options.
    parser.add_argument('--mask_init', default='uniform',
                        choices=['1s', 'uniform', 'weight_based_1s'],
                        help='Type of mask init')
    parser.add_argument('--mask_scale', type=float, default=0.001,
                        help='Mask initialization scaling')

    # parser.add_argument('--mask_init', default='1s',
    #                     choices=['1s', 'uniform', 'weight_based_1s'],
    #                     help='Type of mask init')
    # parser.add_argument('--mask_scale', type=float, default=0,
    #                     help='Mask initialization scaling')
                        
    parser.add_argument('--mask_scale_gradients', type=str, default='none',
                        choices=['none', 'average', 'individual'],
                        help='Scale mask gradients by weights')
    parser.add_argument('--threshold_fn', default='sigmoid',
                        choices=['binarizer', 'ternarizer', 'sigmoid'],
                        help='Type of thresholding function') 

    # other arguments
    # parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    # parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='default false, verbose print')
    parser.add_argument('--reverse', action='store_true', help='default false')
    parser.add_argument('--seed', type=int, default=10, help='random seed (default: 1)')
    parser.add_argument('--gc', type=float, default=1, help='FedAvg: model updates compression keeping rate(default: 1)')
    parser.add_argument('--tradeoff_lambda', type=float, default=0.5, help='lambda parameter')
    parser.add_argument('--target', nargs='+',default=['Heavy_Makeup' 'Male'])
    args = parser.parse_args()
    return args
