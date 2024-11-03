from __future__ import division
from collections import OrderedDict
import math
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch.utils.data import ConcatDataset
import torchvision
from inversefed.nn.models import ConvNet
from models.Update import LocalUpdate as FedAVG
from models.Nets import VGG11, VGG16, AlexNet, CNNLfw, CNNMnist, CNNMotion, fc1
from motion_pia_ldp import load_motion_dataset
from pia import pia_all, pia_mask, pia_per_layer
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_extr_noniid

import os
from torch.utils.data import DataLoader
import copy
import numpy as np
from torchvision import transforms
import torch
from models.dataset import CelebA_property, cifar_noniid, lfw_property, Motionsense, Adult
from utils.options import args_parser
from models.Manager import LocalUpdate

from models.Fed import *
from models.test import test_mnist
import networks as net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.defense = None  #'compensate'## 'outpost' ##'ldp'#'soteria'# ##########
    args.dp_mechanism = 'gauss' if args.defense=='ldp' else None
    args.epsilon = 1 if args.defense is not None else None  ####None  # 1##[40,0.5]
    args.frac = 1#0.3  ###0.1#####client num
    args.no_mask = False  #  True  ##### #是否用fedmask训练
    args.num_users = 2#0#100####0  # 0  ## 2  #200#
    args.epochs = 1#0#00########100## # ###50  # 30
    args.local_ep = 100###20##1# ###5####15##100  #70  # 5#10
    args.EPS = 1###0.3#0.5###0.5###0.1  #######0######0.2##0.1 0.5 ### 0.7#####0.2###top k  0.1  # 0.5  # 0.05  # 0.7##.7  # mask updating threshold 0.9 # 99%更新，1%不变

    args.thrs = 0.5 if not args.no_mask else 0.5
    args.prop_rate = 1  #0.2#########3


    args.num_items_train = 40###150  # 100##args.num_samples  # 20, number of local data size
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    ###########DEFINE Dataset and Model
    args.data_name = 'lfw'#'CelebA'#'mnist'  #'adult' #'cifar10'###  ####
    args.model = 'vgg16'  #'convnet'##### 'cnnlfw'# 'cnn'#'fc'  #### ##'resnet'##### ###
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    args.noise_free = [0]  # 1-N随便选# 设置不加噪的轮次,若为0则每一轮都加噪
    args.delta = 0.00001
    args.clipthr = 5  # C

    # epsilon = 0  # 2 LDP epsilon
    args.layer_wise = False
    args.iid = True  # False#
    args.train_bn = True  #False  #
    args.poison = 0
    args.save = False  #####True#
    selected = False  ##args.selected #True  #######
    # args.tau = 5  # 10

    dissimilarity_layer_wise = [0.972057997,
                                0.965599689,
                                0.961398552,
                                0.960237005,
                                0.958479346,
                                0.956573337,
                                0.956042004,
                                0.954685898,
                                0.951463586,
                                0.949833316,
                                0.949652603,
                                0.94506566,
                                0.938142971,
                                0.800167648,
                                0.83164479,
                                0.948955599
                                ]
    prop = []
    train_data_list = []

    if args.data_name == 'CelebA' or args.data_name == 'lfw':  # 指定数据集
        args.num_items_train = 100##150  # 300 args.num_samples  # 20, number of local data size
        if args.model=='cnnlfw':
            transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        elif args.model=='convnet':
            transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        else:
            transform = transforms.Compose([

                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        args.num_classes = 2
        target = 'Smiling'#args.target[0]
        privacy = 'Male'#args.target[1]  # attr1[5]  #attr1[-2]  # #attr2[0]# attr1[1]#attr1[6] # #指定需要保护的敏感属性
        args.privacy = privacy
        if args.data_name == 'lfw':
            privacy = 'Wavy_Hair'
            if args.model == 'alexnet':
                args.fc_input = 4608
            else:
                args.fc_input = 25088
            data1 = lfw_property('../DATASET/lfw/lfw-deepfunneled/', label_root='../DATASET/lfw/lfw_attributes.txt',
                                 attr=target, transform=transform, property=privacy, non=1)
            # 获取不具有该隐私属性的数据,non=0
            data2 = lfw_property('../DATASET/lfw/lfw-deepfunneled/', label_root='../DATASET/lfw/lfw_attributes.txt',
                                 attr=target,
                                 transform=transform, property=privacy, non=0)
        elif args.data_name == 'CelebA':
            if args.model == 'alexnet':
                args.fc_input = 2560
            else:
                args.fc_input = 15360

            data1 = CelebA_property('../DATASET/celeba/img_align_celeba/',
                                    label_root='../DATASET/celeba/labels.npy', attr=target,
                                    transform=transform, property=privacy, non=1, iid=True)

            data2 = CelebA_property('../DATASET/celeba/img_align_celeba/',
                                    label_root='../DATASET/celeba/labels.npy', attr=target,
                                    transform=transform, property=privacy, non=0, iid=True)  # eyeglasses
    elif args.data_name == 'adult':
        a = np.load('./datafiles/data/adult.npz')
        x = a['data']
        y = a['label']
        # print(x.shape)###(45222, 105)
        target = 'income'
        privacy = 'male'  # attr1[5]  #attr1[-2]  # #attr2[0]# attr1[1]#attr1[6] # #指定需要保护的敏感属性
        data1 = Adult(x, y, non=0)  # male
        data2 = Adult(x, y, non=1)  # male
        args.num_classes = 2
        args.fc_input = 105
    elif args.data_name == 'mnist':
        args.model = 'cnn'
        args.iid = False  # True  #
        args.n_class = 2
        args.lr_mask = 1e-4##5
        args.lr = 0.001
        args.fc_input = 784
        target = ''
        privacy = 'label_{}'.format(args.n_class)
        args.num_classes = 10
        trans_mnist = transforms.Compose([
            # transforms.Resize([128, 128]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        data1 = torchvision.datasets.MNIST('../DATASET/mnist/', train=True, download=True,
                                           transform=trans_mnist)
        testdataset = torchvision.datasets.MNIST('../DATASET/mnist/', train=False, download=False,
                                                 transform=trans_mnist)
    elif args.data_name == 'cifar10':
        args.model = 'convnet'
        args.num_items_train = 200
        args.lr = 0.0001
        args.lr_mask = 1e-6
        args.iid = False
        args.num_classes = 10
        args.n_class = 2
        target = ''
        privacy = 'label_{}'.format(args.n_class)
        args.fc_input = 8192 if args.model == 'vgg16' else 512 #######
        trans_cifar = transforms.Compose([
            # transforms.Resize([224, 224]),
            # transforms.Resize([128, 128]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        data1 = torchvision.datasets.CIFAR10('../DATASET/cifar10/', train=True, download=False,
                                             transform=trans_cifar)
        testdataset = torchvision.datasets.CIFAR10('../DATASET/cifar10/', train=False, download=False,
                                                   transform=trans_cifar)
    else:
        exit('no such dataset!!')

    args.optim = 'fedavg'##'fedprox'### if args.iid else
    args.mu = 1e-3

    if (args.data_name == 'mnist' or args.data_name == 'cifar10'):  #### and not args.iid
        dict_users, dict_users_test, label_users = mnist_extr_noniid(data1, testdataset, args.num_users + 1, args.n_class,
                                                              args.num_items_train, args.unbalance_rate)
        prop = [1 for _ in range(len(dict_users))]
        attack_type=None
        print('non-iid', args.data_name,prop, (label_users))
    else:  # prop
        attack_type = 'pia'
        for i in range(int(args.num_users / 2)):  # female
            prop.append(1)
            num_p1 = int(args.num_items_train * args.prop_rate)
            num_p0 = args.num_items_train - num_p1
            data1, train_dataset = torch.utils.data.random_split(data1, [len(data1) - num_p1,
                                                                         num_p1])
            if num_p0 > 0:
                data2, train_dataset0 = torch.utils.data.random_split(data2, [len(data2) - num_p0,
                                                                              num_p0])
                train_dataset = ConcatDataset([train_dataset, train_dataset0])
                print('total dataset num', len(train_dataset), 'without property', len(train_dataset0))

            train_data_list.append(train_dataset)
        for i in range(int(args.num_users / 2), int(args.num_users)):  # male
            prop.append(0)
            data2, train_dataset = torch.utils.data.random_split(data2, [len(data2) - args.num_items_train,
                                                                         args.num_items_train])
            # train_data = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
            train_data_list.append(train_dataset)  # 100
        data1, test_dataset1 = torch.utils.data.random_split(data1,
                                                             [len(data1) - args.num_items_train,
                                                              args.num_items_train])
        data2, test_dataset2 = torch.utils.data.random_split(data2,
                                                             [len(data2) - args.num_items_train,
                                                              args.num_items_train])

        testdataset = ConcatDataset([test_dataset1, test_dataset2])

    test_data_loader = DataLoader(testdataset, batch_size=args.bs, shuffle=False)
    print('{} test on {} samples'.format(args.data_name, len(testdataset)))
    # build model
    if args.model == 'cnn':
        if args.no_mask:
            net_glob = CNNMnist(args).to(args.device)
        else:
            net_original = CNNMnist(args).to(args.device)
            net_glob = net.CNNMnistModified(args=args, mask_init=args.mask_init,
                                            mask_scale=args.mask_scale,
                                            threshold_fn=args.threshold_fn,
                                            original=args.no_mask, init=net_original).to(args.device)
            net_glob.to(args.device)
        net_glob_list = [copy.deepcopy(net_glob)]  # , copy.deepcopy(net_glob)]
    elif args.model=='convnet':
        args.affine = True
        if not args.no_mask:
            net_original = ConvNet(args=args,width=64, num_channels=3, num_classes=args.num_classes).to(args.device)
            net_glob = net.ModifiedConvNet(args=args, mask_init=args.mask_init,
                                           mask_scale=args.mask_scale,
                                           threshold_fn=args.threshold_fn,
                                           original=args.no_mask, init=net_original,num_class=args.num_classes).to(args.device)

        else:
            net_glob = ConvNet(width=64, num_channels=3, num_classes=args.num_classes).to(args.device)

    elif args.model == 'vgg16':
        args.affine = True###False
        if args.no_mask:
            args.lr = 1e-4
            print('################normal fedavg###################')
            net_glob = VGG16(args).to(args.device)
        else:
            args.lr_mask = 1e-6
            net_original = VGG16(args).to(args.device)

            net_glob = net.VGG16Modified(args, mask_init=args.mask_init,
                                         mask_scale=args.mask_scale, threshold_fn=args.threshold_fn,
                                         original=args.no_mask, init=net_original).to(args.device)
    elif args.model == 'resnet':
        args.arch = 'resnet18'
        args.lr = 0.00001
        if args.no_mask:
            net_glob = net.ModifiedResNet(args, mask_init=args.mask_init,
                                          mask_scale=args.mask_scale,
                                          threshold_fn=args.threshold_fn,
                                          original=True).to(args.device)  # args.no_mask
        else:
            args.lr_mask = 0.000002
            net_original = net.ModifiedResNet(args, mask_init=args.mask_init,
                                              mask_scale=args.mask_scale,
                                              threshold_fn=args.threshold_fn,
                                              original=True).to(args.device)  # args.no_mask

            net_glob = net.ModifiedResNet(args, mask_init=args.mask_init,
                                          mask_scale=args.mask_scale,
                                          threshold_fn=args.threshold_fn,
                                          original=args.no_mask, init=net_original).to(args.device)
    elif args.model == 'alexnet':
        args.lr = 0.0001
        if args.no_mask:
            net_glob = AlexNet(num_classes=args.num_classes, args=args).to(args.device)
        else:
            args.lr_mask = 0.00001
            net_original = AlexNet(num_classes=args.num_classes, args=args).to(args.device)
            net_glob = net.ModifiedAlexNet(args, mask_init=args.mask_init,
                                           mask_scale=args.mask_scale,
                                           threshold_fn=args.threshold_fn,
                                           original=args.no_mask, init=net_original).to(args.device)


    elif args.model == 'cnnlfw':
        args.lr_mask = 1e-5
        if args.no_mask:
            net_glob = CNNLfw(args).to(args.device)
        else:
            net_original = CNNLfw(args).to(args.device)
            net_glob = net.CNNLfwModified(args=args, mask_init=args.mask_init,
                                      mask_scale=args.mask_scale,
                                      threshold_fn=args.threshold_fn,
                                      original=args.no_mask, init=net_original).to(args.device)
    elif args.model == 'fc':
        if args.no_mask:
            args.lr = 0.0005
            net_glob = fc1(dim_in=args.fc_input, num_classes=args.num_classes).to(args.device)
        else:
            args.lr_mask = 1e-4
            net_original = fc1(dim_in=args.fc_input, num_classes=args.num_classes).to(args.device)
            net_glob = net.fc1Modified(args=args, mask_init=args.mask_init,
                                       mask_scale=args.mask_scale, threshold_fn=args.threshold_fn,
                                       original=args.no_mask, init=net_original).to(args.device)
    else:
        exit('Error: unrecognized model')

    net_glob_list = [copy.deepcopy(net_glob)]  # ,, copy.deepcopy(net_glob) copy.deepcopy(net_glob)]#

    # Initialize with weight based method, if necessary.
    if not args.no_mask and args.mask_init == 'weight_based_1s':
        print('Are you sure you want to try this?')
        # assert args.mask_scale_gradients == 'none'
        # assert not args.mask_scale
        for idx, module in enumerate(net_glob.modules()):
            if 'ElementWise' in str(type(module)):
                weight_scale = module.weight.data.abs().mean()
                module.mask_real.data.fill_(weight_scale)

    # args.prop = prop
    mask_real_tasks = [{}, {}, {}]
    mask_avg_tasks = [{}, {}, {}]

    # mask_real = {}
    assert args.lr_mask and args.lr_mask_decay_every
    assert args.lr_classifier and args.lr_classifier_decay_every

    m = max(int(args.frac * args.num_users), 1)
    idxs_poison_users = []
    if args.poison > 0:
        poison_users = max(int(args.poison * m), 1)
        idxs_poison_users = np.random.choice(range(args.num_users), poison_users, replace=False)  # 用户id列表

    mask_diff = []
    label_prop = []
    mask_diff_test = []
    label_prop_test = []
    if not selected or args.no_mask:
        print('no selecting masks')
        args.k = None
        args.tau = None
    filename = '{}_{}_selected_{}_iid_{}_{}_{}_{}_{}_EPS_{}_{}_{}_poison_{}_frac_{}_prop_rate_{}_local_ep{}_thr{}.txt'.format(
        args.optim,args.no_mask, args.tau, args.iid, args.model, args.data_name, target, privacy, args.EPS, args.defense, args.epsilon,
        args.poison, args.frac, args.prop_rate, args.local_ep,args.thrs)
    with open(filename, 'a+') as f:
        f.write('{}\n'.format(args))
        print('logging to',filename, args)
    acc_test, _, _ = test_mnist(net_glob_list[0], test_data_loader, args, prop[0])
    print('initial test acc', acc_test)
    with open(filename, 'a+') as f:
        print('Round\t{}\tAcc\t{}\t\n'.format(0, acc_test))
        f.write('Round\t{}\tAcc\t{}\t\n'.format(0, acc_test))
    # features = []
    y = []
    for iter in range(1, args.epochs + 1):
        acc_tasks = []
        mask_tasks = []  # client mask update
        # 随机选取本轮更新的设备id
        if m == args.num_users:
            idxs_users = np.arange(0, args.num_users)
        else:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 用户id列表
        print(iter, "轮更新的设备id", idxs_users)
        if not args.no_mask:  ######fedmask
            for task in range(1):
                mask_real = mask_real_tasks[task]  # {},实数mask
                mask_avg = mask_avg_tasks[task]  # {}0,1
                mask_locals = []
                sensitivity = None
                # net_glob = net_glob
                local_test_acc = []
                if iter == 1:  # 保存初始随机化mask
                    mask_ori = {}
                    layerid = []
                    for module_idx, module in enumerate(net_glob_list[task].modules()):
                        if 'ElementWise' in str(type(module)):
                            layerid.append(module_idx)
                            mask_ori[module_idx] = module.mask_real.data.clone()
                            mask_ori[module_idx].fill_(0)
                            mask_ori[module_idx][module.mask_real.data > 0] = 1
                    # layerid = layerid[:3]
                    mask_avg_tasks[task] = mask_ori
                    if selected:
                        args.k = {i: j / args.tau for i, j in zip(layerid, dissimilarity_layer_wise)}
                    print(layerid,'k', args.k)

                for idx in idxs_users:  # prop[idx] = task
                    # y.append(str(sorted(label_users[idx])))
                    train_model = copy.deepcopy(net_glob_list[task])
                    if iter > 1:
                        if idx in mask_real.keys():
                            # print(idx,'initialize real-valued mask')
                            for module_idx, module in enumerate(train_model.modules()):
                                if 'ElementWise' in str(type(module)):
                                    module.mask_real.data = mask_real[idx][module_idx]  #####恢复客户端本身的mask_real值
                                    module.mask_real.data[mask_avg[module_idx] == 1] = torch.abs(
                                        module.mask_real.data[mask_avg[module_idx] == 1])
                                    module.mask_real.data[mask_avg[module_idx] == 0] = -1 * torch.abs(
                                        module.mask_real.data[mask_avg[module_idx] == 0])
                                    # print('neq=0??', (module.mask_real.data != mask_real[idx][module_idx]).sum().item())
                                    # print(module.mask_real.data[mask_avg[module_idx] == 0],'after positive')
                                elif args.train_bn and 'BatchNorm' in str(type(module)):
                                    # print('local updating batch norm layer')
                                    if module.weight is not None:
                                        module.weight.data = mask_real[idx][module_idx]['weight']
                                        module.bias.data = mask_real[idx][module_idx]['bias']
                                    if module.running_var is not None:
                                        module.running_mean = mask_real[idx][module_idx]['running_mean']
                                        module.running_var = mask_real[idx][module_idx]['running_var']
                                    # Masking will be done.
                    if not args.iid:  # non-iid 需要特别指定客户端的测试集    copy.deepcopy
                        local = LocalUpdate(args=args, model=train_model, dataset=data1, idxs=dict_users[idx],
                                            prop=prop[idx],
                                            checkpoints=f'./mask_ckp/{args.data_name}_{args.model}_{prop[idx]}/',
                                            dtest=testdataset, attack_type=attack_type,
                                            idxs_test=dict_users_test[idx], update_gradient=args.EPS)
                    else:
                        local = LocalUpdate(args=args, model=train_model, dataset=train_data_list[idx], idxs=None,
                                            prop=prop[idx],
                                            checkpoints=f'./mask_ckp/{args.data_name}_{args.model}_{target}_{privacy}/',
                                            dtest=testdataset, attack_type=attack_type,
                                            update_gradient=args.EPS)
                    mask_real_local, masks, sensitivity, local_acc = local.train(args.local_ep, epsilon=args.epsilon,
                                                                                 sen=sensitivity,
                                                                                 layer_wise=args.layer_wise,
                                                                                 save=args.save,
                                                                                 savename=f'{prop[idx]}_ep_{iter}_client_{idx}',
                                                                                 mask_ori=mask_ori,return_updates=False)
                    mask_locals.append(copy.deepcopy(masks))
                    local_test_acc.append(local_acc)
                    mask_real[idx] = mask_real_local
                mask_real_tasks[task] = mask_real
                mask_tasks.append(mask_locals)

                net_glob_list[task], _, mask_avg, _, _ = FedCompare(net_glob_list[task], mask_locals, prop,###*(1+iter/args.epochs)
                                                                    mask_ori, epsilon=args.epsilon, layer_wise=args.layer_wise,threshold=args.thrs,return_updates=False)
                mask_diff.extend(pia_difference(net_glob_list[task], mask_locals, mask_ori,return_update=False,args=args,iter=iter))  # [client1{layer1:[],layer2:[],...},client2{},...]
                label_prop.extend(prop)
                mask_avg_tasks[task] = mask_avg
                # update mask_ori as the avg mask of last round
                for module_idx in mask_ori.keys():
                    mask_ori[module_idx][mask_avg[module_idx] != 2] = mask_avg[module_idx][mask_avg[module_idx] != 2]
                    # print('neq num: ',(mask_ori[module_idx][mask_avg[module_idx] != 2]!=mask_avg[module_idx][mask_avg[module_idx] != 2]).sum().item())
                for module_idx, module in enumerate(net_glob_list[task].modules()):
                    if module_idx in mask_ori.keys():
                        module.mask_real.data[mask_ori[module_idx] == 1] = 0.001
                        module.mask_real.data[mask_ori[module_idx] == 0] = -0.001

                acc_test, _, _ = test_mnist(net_glob_list[task], test_data_loader, args, prop[idx])
                acc_tasks.append(acc_test)


        else:  ######FedAVG
            for task in range(1):
                if iter == 1:
                    w_avg = copy.deepcopy(net_glob_list[task].state_dict())
                    layerid = []
                    w_avg.keys()
                    for module, _ in net_glob_list[task].named_parameters():
                        print(module)
                        if 'weight' in module and 'batch' not in module:
                            layerid.append(module)
                    # layerid=layerid[24:]
                    print(layerid)
                local_test_acc = []
                w_locals, loss_locals = [], []
                poison_label = []
                for idx in idxs_users:
                    if idx in idxs_poison_users:
                        poisoning = True
                        poison_label.append('malicious')
                    else:
                        poisoning = False
                        poison_label.append('benign')
                    if not args.iid:  # non-iid 需要特别指定客户端的测试集    copy.deepcopy
                        local = FedAVG(args=args, dataset=data1, idxs=dict_users[idx],
                                       prop=prop[idx],
                                       checkpoints='./no_mask_{}_{}_{}_lr_{}_{}_chosen_{}_{}_eps_{}_c_{}_no_noise_{}/'.format(
                                           args.data_name, '', '', args.lr, args.num_users, m, args.dp_mechanism,
                                           args.epsilon,
                                           args.clipthr,
                                           args.noise_free), dtest=testdataset, attack_type=attack_type,
                                idxs_test=dict_users_test[idx], update_gradient=args.EPS, poisoning=poisoning)
                    else:  # iid
                        local = FedAVG(args=args, dataset=train_data_list[idx], idxs=None,
                                       prop=prop[idx],checkpoints=f'./mask_ckp/{args.data_name}_{args.model}_{target}_{privacy}/',
                                       dtest=testdataset, attack_type=attack_type,
                                       update_gradient=args.EPS, poisoning=poisoning)
                    w, local_acc, grad, grad_train = local.train(net=copy.deepcopy(net_glob_list[task]),
                                                                 global_epoch=iter, return_updates=True)
                    mask_diff_test.append(grad)
                    label_prop_test.append(prop[idx])
                    mask_diff.append(grad_train)  ###epoch-wise
                    label_prop.append(prop[idx])
                    # if iter < 20:
                    #     property_tr_list.append(prop[idx])
                    #     grad_tr_list.append(grad_train)
                    w_locals.append(w)
                    local_test_acc.append(local_acc)
                if args.poison > 0:
                    similarity = compute_similarity_poison_fedavg(w_avg, w_locals,
                                                                  poison_label)  ###similarity = {'malicious': [],'benign':[]}
                w_avg = FedAvg(w_locals, return_updates=True, w_global=net_glob_list[
                    task].state_dict())  ###  poison_label=poison_label  ###########  Aggregation
                net_glob_list[task].load_state_dict(w_avg)
                acc_test, _, _ = test_mnist(net_glob_list[task], test_data_loader, args, prop[idx])
                acc_tasks.append(acc_test)

        with open(filename, 'a+') as f:
            print('Round\t{}\tAcc\t{}\tLocal Acc\t{}\n'.format(iter, acc_tasks[0], np.mean(local_test_acc)))

            f.write('Round\t{}\tAcc\t{}\tLocal Acc\t{}\n'.format(iter, acc_tasks[0], np.mean(local_test_acc)))
    acc=None
    layeracc=None
    if args.data_name not in ['mnist','cifar10']:
        if not args.no_mask:  ##fedmask
            acc = pia_mask(layerid, None, None, args=args, target=target, privacy=privacy, trainset=mask_diff,
                           train_prop=label_prop, iter=iter, clients_num=args.num_users, m=m,labels=y)
        else:######mask_diff_test, label_prop_test,
            acc = pia_all(layerid,mask_diff_test, label_prop_test,  args=args, target=target, privacy=privacy,
                          trainset=mask_diff,train_prop=label_prop, iter=iter, clients_num=args.num_users, m=m)

            # layeracc=pia_per_layer(layerid,mask_diff_test, label_prop_test,  args=args, target=target, privacy=privacy,
            #               trainset=mask_diff,train_prop=label_prop, iter=iter, clients_num=args.num_users, m=m)



        with open(filename, 'a+') as f:
            f.write('PIA Acc\t{}\t\n'.format(acc))
            print('PIA Acc\t{}\n'.format(acc))
            print('PIA Layer Acc\t{}\n'.format(layeracc))

            f.write('PIA Layer Acc\t{}\t\n'.format(layeracc))
    else:
        acc = pia_mask(layerid, None, None, args=args, target=target, privacy=privacy, trainset=mask_diff,
                       train_prop=label_prop, iter=iter, clients_num=args.num_users, m=m, labels=y)
