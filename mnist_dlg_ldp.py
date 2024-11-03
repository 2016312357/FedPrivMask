from __future__ import division
from __future__ import division
from collections import OrderedDict
import math

# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
from models.Update import LocalUpdate as FedAVG
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset
import torchvision
from dlg import reconstruct, reconstruct_fedavg
import models

from models.Nets import VGG11, VGG16, AlexNet, CNNLfw, CNNMnist  ##, fc1
from sensitivity import compute_sens
# from pia import pia_all
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_extr_noniid

matplotlib.use('Agg')
import os
from torch.utils.data import DataLoader
import copy
import numpy as np
from torchvision import transforms
import torch
from models.dataset import CelebA_property, cifar_noniid, lfw_property
from utils.options import args_parser
from models.Manager import LocalUpdate
from models.Fed import *  # FedAvg, FedAvgMask, FedCompare, compute_dissimilarity, compute_similarity_poison
from models.test import test_mnist
import networks as net


def plot(tensor, tensor1):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    tensor1 = tensor1.clone().detach()
    tensor1.mul_(ds).add_(dm).clamp_(0, 1)
    print('plotting', tensor.shape)  ###[8, 3, 32, 32]
    if tensor.shape[0] == 1:
        plt.subplot(2, 1, 1)
        plt.imshow(tensor1[0].permute(1, 2, 0).cpu(), cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(tensor[0].permute(1, 2, 0).cpu(), cmap='gray')

        # return plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(2, tensor.shape[0], figsize=(12, 20))
        for i, im in enumerate(tensor1):
            axes[1][i].imshow(im.permute(1, 2, 0).cpu(), cmap='gray')
        for i, im in enumerate(tensor):
            axes[0][i].imshow(im.permute(1, 2, 0).cpu(), cmap='gray')


from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def dlg_attack(train_model, args, grad_avg, trainset, iter, idx, masks, mask_ori, dm, ds, fedavg=True, save_image=True,
               save_path='results/'):
    config = dict(signed=True, boxed=True, cost_fn='sim', indices='def', weights='equal',
                  lr=0.0005, optim='adam', restarts=1, max_iterations=24000,  # 20000 10000
                  total_variation=1e-6, init='randn', filter='none', lr_decay=True, scoring_choice='loss')
    import inversefed
    # setup = inversefed.utils.system_startup()
    # defs = inversefed.training_strategy('conservative', lr=0.01, epochs=4)  ###def.batchsize=64
    # num_images=1
    # loss_fn, trainloader, validloader = inversefed.construct_dataloaders('MNIST', defs, '../DATASET/mnist/')
    # # loss_fn, trainloader, validloader = inversefed.construct_dataloaders('CIFAR10', defs, '../DATASET/cifar10/')
    # ground_truth, labels = [], []
    # target_id = 35#30 #37#27#25# # choosen randomly ... just whatever you want
    # while len(labels) < num_images:
    #     img, label = validloader.dataset[target_id]
    #     # img, label = trainloader.dataset[target_id]
    #     target_id += 1
    #     labels.append(torch.as_tensor((label,), device=args.device))
    #     ground_truth.append(img.to(**setup))
    # ground_truth = torch.stack(ground_truth)
    # labels = torch.cat(labels)
    ground_truth, labels = [], []
    target_id = 0
    # num_images = args.n_class
    while target_id < len(trainset) and len(labels) < args.n_class:
        ground_truth_, label = trainset[target_id]  ###已经经过transform操作
        if label not in labels:
            ground_truth.append(ground_truth_.to(args.device))
            labels.append(torch.as_tensor((label,), device=args.device))
        target_id += 1
    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    img_shape = (ground_truth.shape[1], ground_truth.shape[2], ground_truth.shape[3])
    if fedavg:
        parameters = list()
        print('directly attack the model updates', ground_truth.shape)
        if not args.no_mask:
            original_parameters = inversefed.reconstruction_algorithms.loss_steps(train_model, ground_truth, labels,
                                                                                  lr=args.lr, local_steps=0,
                                                                                  use_updates=False)  ###real-valued
        index = 0
        for g in masks.keys():
            grad = masks[g].clone()  # binary mask
            if not args.no_mask:
                ori_grad = original_parameters[index]  # .clone()  # real-valued
                index += 1
                grad.fill_(0)
                grad[masks[g] == 1] = ori_grad[masks[g] == 1]  ###negative
                grad[masks[g] == -1] = torch.abs(ori_grad[masks[g] == -1])
                # print(grad[masks[g] == -1],'all positive??')
                # print(grad[masks[g]==1],'all negative??')
                # grad[masks[g]==1]=-0.001#/args.lr_mask##args.lr ##local updates of real-valued mask
                # grad[masks[g]==-1]=0.001#/args.lr_mask
            elif 'running' in g or 'batches' in g:  ###fedavg, batchnorm layer only weight and bias can be used
                print('no', g)
                continue
            parameters.append(grad.detach())
        print('no mask? ', args.no_mask, 'using param length', len(parameters), len(masks.keys()))

    else:
        print('simulate the local updates', ground_truth.shape)
        train_model.zero_grad()
        train_model.train()
        input_parameters = inversefed.reconstruction_algorithms.loss_steps(train_model, ground_truth, labels,
                                                                           lr=args.lr, local_steps=args.local_ep,
                                                                           use_updates=True)
        parameters = [p.detach() for p in input_parameters]

    if args.no_mask:
        rec_machine = inversefed.FedAvgReconstructor(train_model, (dm, ds), args.local_ep, 0.01, config,
                                                     num_images=len(ground_truth),
                                                     use_updates=True)  # ,batch_size=len(ground_truth)
    else:
        rec_machine = inversefed.FedAvgReconstructor(train_model, (dm, ds), args.local_ep, 0.01, config,
                                                     num_images=len(ground_truth),
                                                     use_updates=True)  # , batch_size=len(ground_truth)
    output, stats = rec_machine.reconstruct(parameters, labels, img_shape=img_shape, ground_truth=ground_truth)

    test_mse = (output.detach() - ground_truth).pow(2).mean()
    feat_mse = (torch.nn.functional.softmax(train_model(output.detach()), dim=-1) - torch.nn.functional.softmax(
        train_model(ground_truth), dim=-1)).pow(2).mean()
    test_psnr = inversefed.metrics.psnr(output.detach(), ground_truth, factor=1 / ds)
    test_ssim = ssim(ground_truth, output)

    os.makedirs(f'{save_path}/', exist_ok=True)
    plot(output, ground_truth)
    plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
              f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4f} | SSIM:{test_ssim:2.4f}")
    plt.savefig(f'{save_path}/{args.data_name}_round_{iter}_client_{idx}_{target_id}.png')
    print('saving to', save_path, f'/{args.data_name}_round_{iter}_client_{idx}_{target_id}.png')
    exit(0)


class DatasetSplit(torch.utils.data.Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset

        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label  # torch.tensor(image), torch.tensor(label)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from PIL import Image

cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.dp_mechanism = None  ## 'gauss' #  None 'gauss'  # #'gauss'  # None#'gauss'  # 'gauss'  ### gauss or laplace ###
    # 设置不加噪的轮次,若为0则每一轮都加噪
    args.noise_free = [0]  # 1-N随便选
    args.epsilon = 20
    args.delta = 0.0001
    args.clipthr = 0.5  # C
    args.num_items_train = 30  # 100# 5#100  args.num_samples  # 20, number of local data size
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    args.data_name = 'cifar10'# 'mnist'  #'cifar100'###'stl' #####
    args.model ='vgg16'#'resnet'#'cnn'  #
    args.k = None
    args.EPS = 0  # 0.7#.8#.9##.7#0.7#0.3###fedavg梯度更新稀疏化/mask updating threshold 0.9 # 99%更新，1%不变
    epsilon = 0  # LDP epsilon
    args.local_ep = 5##0  ####15##50#10#5#####100#
    args.num_users = 100###100  ###10#
    args.frac = 0.1##0.2
    args.epochs = 20  ##30#100#2#
    args.n_class = 2  # 5
    args.iid = True  # False#cifar10True  #
    args.optim = 'fedavg' if args.iid else 'fedprox'
    args.pretrain = False  # True #预训练
    args.no_mask = True  ### False ####是否用fedmask训练
    args.defense = ''  #'mask'#'compensate' #'soteria'# ###''####''##'compensate'###
    args.save=False####True
    if args.defense == 'mask':
        args.no_mask = False
    args.train_bn = True  # False#
    perform_attack = False  #######False#########

    args.poison = 0  # 0.25#.4#.2#.4#0.2##.2#malicious clients rate

    if args.data_name == 'mnist':
        args.lr_mask = 0.0001
        args.lr=0.001
        args.fc_input = 784
        target = ''
        privacy = 'label_{}'.format(args.n_class)
        args.num_classes = 10
        # mnist_mean = (0.1307,)
        # mnist_std = (0.3081,)
        trans_mnist = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mnist_mean, mnist_std)
        ])
        # dm = torch.as_tensor(mnist_mean[0])
        # ds = torch.as_tensor(mnist_std[0])
        data1 = torchvision.datasets.MNIST('../DATASET/mnist/', train=True, download=True,
                                           transform=trans_mnist)
        testdata = torchvision.datasets.MNIST('../DATASET/mnist/', train=False, download=False,
                                              transform=trans_mnist)
    elif args.data_name == 'cifar10':
        args.num_items_train = 200
        args.lr = 0.00001
        target = ''
        privacy = 'label_{}'.format(args.n_class)
        args.num_classes = 10
        args.fc_input = 512  #######
        # transform = transforms.Compose(, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #     [transforms.Resize([224, 224]),  transforms.ToTensor()])

        cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]  # [0,0,0]#
        cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]  # [1,1,1]#
        trans_cifar = transforms.Compose([# transforms.Resize([224, 224]),
            transforms.Resize([128, 128]),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std)])
        dm = torch.as_tensor(cifar10_mean, device=args.device)[:, None,
             None]  ##array([[[0.4914672374725342]],[[0.4822617471218109]],[[0.4467701315879822]]])
        ds = torch.as_tensor(cifar10_std, device=args.device)[:, None, None]
        data1 = torchvision.datasets.CIFAR10('../DATASET/cifar10/', train=True, download=False,
                                             transform=trans_cifar)
        testdata = torchvision.datasets.CIFAR10('../DATASET/cifar10/', train=False, download=False,
                                                transform=trans_cifar)
    elif args.data_name == 'cifar100':
        args.num_items_train = 1000
        args.lr = 0.00001
        target = ''
        privacy = 'label_{}'.format(args.n_class)
        args.num_classes = 100
        args.fc_input = 512
        trans_cifar = transforms.Compose([
            transforms.Resize([224, 224]),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std)])
        # dm = torch.as_tensor(imagenet_mean, device=args.device)[:, None,
        #      None]  ##array([[[0.4914672374725342]],[[0.4822617471218109]],[[0.4467701315879822]]])
        # ds = torch.as_tensor(imagenet_std, device=args.device)[:, None, None]
        dm = torch.as_tensor(cifar100_mean, device=args.device)[:, None,
             None]  ##array([[[0.4914672374725342]],[[0.4822617471218109]],[[0.4467701315879822]]])
        ds = torch.as_tensor(cifar100_std, device=args.device)[:, None, None]
        data1 = torchvision.datasets.CIFAR100('../DATASET/cifar100/', train=True, download=True,
                                              transform=trans_cifar)
        testdata = torchvision.datasets.CIFAR100('../DATASET/cifar100/', train=False, download=True,
                                                 transform=trans_cifar)
    elif args.data_name == 'stl':
        args.num_items_train = 200
        args.lr = 0.00001
        target = ''
        privacy = 'label_{}'.format(args.n_class)
        args.num_classes = 10
        args.fc_input = 512  #######
        # transform = transforms.Compose(, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #     [transforms.Resize([224, 224]),  transforms.ToTensor()])

        trans_cifar = transforms.Compose([
            # transforms.Resize([32, 32]),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)])
        # dm = torch.as_tensor(imagenet_mean, device=args.device)[:, None,
        #      None]  ##array([[[0.4914672374725342]],[[0.4822617471218109]],[[0.4467701315879822]]])
        # ds = torch.as_tensor(imagenet_std, device=args.device)[:, None, None]
        dm = torch.as_tensor(cifar100_mean, device=args.device)[:, None,
             None]  ##array([[[0.4914672374725342]],[[0.4822617471218109]],[[0.4467701315879822]]])
        ds = torch.as_tensor(cifar100_std, device=args.device)[:, None, None]
        data1 = torchvision.datasets.CIFAR100('../DATASET/cifar100/', train=True, download=True,
                                              transform=trans_cifar)
        testdata = torchvision.datasets.CIFAR100('../DATASET/cifar100/', split='test', download=True,
                                                 transform=trans_cifar)

    if args.data_name == 'mnist' or args.data_name == 'cifar10' or args.data_name == 'cifar100':
        train_data_list = []
        if args.iid:
            prop = []
            for i in range(int(args.num_users)):  # female
                prop.append(1)  # 0:奇偶性， 1：10分类,  2:大于等于5
                data1, train_dataset = torch.utils.data.random_split(data1, [len(data1) - args.num_items_train,
                                                                             args.num_items_train])
                train_data_list.append(train_dataset)
            # traindataset = train_data_list[-1]
            # train_loader = DataLoader(traindataset, batch_size=args.local_bs, shuffle=True)
            testdataset = testdata
        else:  # non-iid
            # sample data for users        
            dict_users, dict_users_test, prop = mnist_extr_noniid(data1, testdata, args.num_users + 1, args.n_class,
                                                                  args.num_items_train, args.unbalance_rate)
            print('non-iid mnist', prop)
            if not args.pretrain:
                ### train data is non-iid, easy for dlg attack
                testdataset = testdata
                train_loader = DataLoader(DatasetSplit(data1, dict_users[args.num_users]), batch_size=args.local_bs,
                                          shuffle=True)
            else:
                #### train data is iid, same distribution as local data
                traindataset, testdataset = torch.utils.data.random_split(testdata, [int(len(testdata) * 0.05),
                                                                                     len(testdata) - int(
                                                                                         len(testdata) * 0.05)])
                # [args.num_items_train,
                #  len(testdata) - args.num_items_train])  # [int(len(testdata) * 0.3),# len(testdata) - int(len(testdata) * 0.3)])
                train_loader = DataLoader(traindataset, batch_size=args.local_bs, shuffle=True)
    test_data_loader = DataLoader(testdataset, batch_size=args.bs, shuffle=False)
    print('{} test on {} samples'.format(args.data_name, len(testdataset)))

    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
        net_glob_list = [net_glob, copy.deepcopy(net_glob)]  # , copy.deepcopy(net_glob)]
    # elif args.model == 'mlp':
    #     if args.no_mask:
    #         net_glob = fc1(args.num_classes).to(args.device)
    #     else:
    #         net_original = fc1(args.num_classes).to(args.device)
    #         net_glob = net.fc1Modified(args=args, mask_init=args.mask_init,
    #                                mask_scale=args.mask_scale, threshold_fn=args.threshold_fn,
    #                                original=args.no_mask, init=net_original).to(args.device)
    #         net_glob.to(args.device)
    #         acc_test, loss_test, _ = test_mnist(net_glob, test_data_loader, args)
    #         print("original pretrain Testing accuracy: {:.2f}, loss per sample: {:.2f}".format(acc_test, loss_test))
    #
    #     net_glob_list = [net_glob, copy.deepcopy(net_glob)]#, copy.deepcopy(net_glob)]
    elif args.model == 'vgg16':
        if args.no_mask:
            net_glob = VGG16(args).to(args.device)
        else:
            args.lr_mask = 0.000001  ########
            net_original = VGG16(args).to(args.device)

            acc_test, loss_test, _ = test_mnist(net_original, test_data_loader, args, prop=0)
            print("original pretrain Testing accuracy: {:.2f}, loss per sample: {:.2f}".format(acc_test, loss_test))
            net_glob = net.VGG16Modified(args, mask_init=args.mask_init,
                                         mask_scale=args.mask_scale,
                                         threshold_fn=args.threshold_fn,
                                         original=args.no_mask, init=net_original).to(args.device)
            net_glob.to(args.device)
        net_glob_list = [net_glob, copy.deepcopy(net_glob)]

    elif args.model == 'resnet':
        args.arch = 'resnet50'
        if args.no_mask:
            net_glob = net.ModifiedResNet(args, mask_init=args.mask_init,
                                              mask_scale=args.mask_scale,
                                              threshold_fn=args.threshold_fn,
                                              original=True).to(args.device)  # args.no_mask

            acc_test, loss_test, _ = test_mnist(net_glob, test_data_loader, args)
            print("original pretrain Testing accuracy: {:.2f}, loss per sample: {:.2f}".format(acc_test, loss_test))
        else:
            net_original = net.ModifiedResNet(args, mask_init=args.mask_init,
                                              mask_scale=args.mask_scale,
                                              threshold_fn=args.threshold_fn,
                                              original=True).to(args.device)  # args.no_mask

            net_glob = net.ModifiedResNet(args, mask_init=args.mask_init,
                                          mask_scale=args.mask_scale,
                                          threshold_fn=args.threshold_fn,
                                          original=args.no_mask, init=net_original).to(args.device)
        net_glob_list = [net_glob, copy.deepcopy(net_glob)]

    elif args.model == 'alexnet':
        net_original = AlexNet(num_classes=args.num_classes).to(args.device)

        acc_test, loss_test, _ = test_mnist(net_original, test_data_loader, args)
        print("original pretrain Testing accuracy: {:.2f}, loss per sample: {:.2f}".format(acc_test, loss_test))

        net_glob = net.ModifiedAlexNet(args, mask_init=args.mask_init,
                                       mask_scale=args.mask_scale,
                                       threshold_fn=args.threshold_fn,
                                       original=args.no_mask, init=net_original).to(args.device)
        net_glob_list = [copy.deepcopy(net_glob), copy.deepcopy(net_glob)]

    elif args.model == 'cnnlfw':
        net_original = CNNLfw(args).to(args.device)
        acc_test, loss_test, _ = test_mnist(net_original, test_data_loader, args)
        print("original pretrain Testing accuracy: {:.2f}, loss per sample: {:.2f}".format(acc_test, loss_test))

        net_glob = net.CNNLfwModified(args=args, mask_init=args.mask_init,
                                      mask_scale=args.mask_scale,
                                      threshold_fn=args.threshold_fn,
                                      original=args.no_mask, init=net_original).to(args.device)


    else:
        exit('Error: unrecognized model')

        # Initialize with weight based method, if necessary.
    if not args.no_mask and args.mask_init == 'weight_based_1s':
        print('Are you sure you want to try this?')
        # assert args.mask_scale_gradients == 'none'
        # assert not args.mask_scale
        for idx, module in enumerate(net_glob.modules()):
            if 'ElementWise' in str(type(module)):
                weight_scale = module.weight.data.abs().mean()
                module.mask_real.data.fill_(weight_scale)
    mask_real_tasks = [{}, {}, {}]
    mask_avg_tasks = [{}, {}, {}]

    assert args.lr_mask and args.lr_mask_decay_every
    assert args.lr_classifier and args.lr_classifier_decay_every
    with open('fedavg_{}_iid_{}_{}_{}_{}_{}_EPS_{}_ldp_{}_poison_{}_{}.txt'.format(args.no_mask, args.iid, args.model,
                                                                                args.data_name, target, privacy,
                                                                                args.EPS, epsilon, args.poison,args.defense),
              'a+') as f:
        f.write('{}\n'.format(args))
    m = max(int(args.frac * args.num_users), 1)
    idxs_poison_users = []
    if args.poison > 0:
        poison_users = max(int(args.poison * m), 1)
        idxs_poison_users = np.random.choice(range(args.num_users), poison_users, replace=False)  # 用户id列表

    for iter in range(1, args.epochs + 1):
        # if iter%50==0:
        #     args.lr_mask/=2
        if m == args.num_users:
            idxs_users = np.arange(0, args.num_users)
        else:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 用户id列表
        print(iter, "轮更新的设备id", idxs_users, 'poisoned user', idxs_poison_users)
        acc_tasks = []

        if not args.no_mask:  ######fedmask
            print('using mask-based training')
            mask_tasks = []
            for task in range(1, 2):
                mask_real = mask_real_tasks[task]  # {},实数mask
                mask_avg = mask_avg_tasks[task]  # {}0,1
                mask_locals = []
                sensitivity = None
                local_test_acc = []
                if iter == 1:
                    mask_ori = {}  # 保存第一轮的随机初始mask
                    for module_idx, module in enumerate(net_glob_list[task].modules()):
                        if 'ElementWise' in str(type(module)):
                            mask_ori[module_idx] = module.mask_real.data.clone()
                            mask_ori[module_idx].fill_(0)
                            mask_ori[module_idx][module.mask_real.data > 0] = 1
                    mask_avg_tasks[task] = mask_ori
                poison_label = []
                for idx in idxs_users:
                    if idx in idxs_poison_users:
                        poisoning = True
                        poison_label.append('malicious')
                    else:
                        poisoning = False
                        poison_label.append('benign')

                    prop[idx] = task
                    train_model = copy.deepcopy(net_glob_list[task])
                    if iter > 1:
                        if idx in mask_real.keys():  ##如若从未选中过该客户端，则和当前全局模型保持一致
                            # print(idx,'initialize real-valued mask',args.mask_init)
                            for module_idx, module in enumerate(train_model.modules()):
                                if 'ElementWise' in str(type(module)):
                                    ###不同的initialize方式，保留上一轮local model的real_mask绝对值，只根据global mask改变符号
                                    output = mask_real[idx][module_idx].clone()
                                    output[mask_avg[module_idx] == 1][output[mask_avg[module_idx] == 1] < 0] *= -1
                                    output[mask_avg[module_idx] == 0][output[mask_avg[module_idx] == 0] > 0] *= -1
                                    module.mask_real.data = output
                                elif args.train_bn and 'BatchNorm' in str(type(module)):
                                    print('local updating batch norm layer')
                                    module.weight.data = mask_real[idx][module_idx]['weight']
                                    module.bias.data = mask_real[idx][module_idx]['bias']
                                    module.running_mean = mask_real[idx][module_idx]['running_mean']
                                    module.running_var = mask_real[idx][module_idx]['running_var']

                    if not args.iid:  # non-iid 需要特别指定客户端的测试集    copy.deepcopy
                        local = LocalUpdate(args=args, model=train_model, dataset=data1, idxs=dict_users[idx],
                                            prop=prop[idx],
                                            checkpoints=f'./mask_ckp/{args.data_name}_{args.model}_{prop[idx]}/',

                                            # checkpoints='./{}_{}_{}_lr_{}_{}_chosen_{}_{}_eps_{}_c_{}_no_noise_{}/'.format(
                                            #     args.data_name, '', '', args.lr, args.num_users, m, args.dp_mechanism,
                                            #     args.epsilon,
                                            #     args.clipthr,
                                            #     args.noise_free), 
                                            dtest=testdata, attack_type='pia',
                                            idxs_test=dict_users_test[idx], update_gradient=args.EPS,
                                            poisoning=poisoning)
                    else:  # iid  
                        local = LocalUpdate(args=args, model=train_model, dataset=train_data_list[idx], idxs=None,
                                            prop=prop[idx],
                                            checkpoints=f'./mask_ckp/{args.data_name}_{args.model}_{prop[idx]}/',
                                            dtest=testdata, attack_type='pia',
                                            update_gradient=args.EPS,
                                            poisoning=poisoning)  # None
                    mask_real_local, masks, grad_avg, local_acc = local.train(args.local_ep, epsilon=epsilon,
                                                                              sen=sensitivity, save=args.save,
                                                                              savename=f'ep_{iter}_{idx}',
                                                                              mask_ori=mask_avg_tasks[task])
                    if perform_attack:
                        dlg_attack(net_glob_list[task], args, grad_avg, local.ldr_train.dataset, iter, idx, masks,
                                   mask_avg_tasks[task], fedavg=True, dm=dm, ds=ds,
                                   save_path='fedavg_{}_iid_{}_{}_{}_{}_{}_EPS_{}_ldp_{}_poison_{}'.format(args.no_mask,
                                                                                                           args.iid,
                                                                                                           args.model,
                                                                                                           args.data_name,
                                                                                                           target,
                                                                                                           privacy,
                                                                                                           args.EPS,
                                                                                                           epsilon,
                                                                                                           args.poison))
                    mask_locals.append(copy.deepcopy(masks))
                    mask_real[idx] = mask_real_local
                    local_test_acc.append(local_acc)
                if args.poison > 0:
                    ###detect and filtering poisoners
                    similarity = compute_similarity_poison_fedmask(mask_avg_tasks[task], mask_locals,
                                                                   poison_label)  ###similarity = {'malicious': [],'benign':[]}

                mask_real_tasks[task] = mask_real
                # mask_tasks.append(mask_locals)
                net_glob_list[task], _, mask_avg, test_masks, _ = FedCompare(net_glob_list[task], mask_locals, prop,
                                                                             mask_ori, epsilon=epsilon,
                                                                             layer_wise=False)  # poison_label=poison_label,
                mask_avg_tasks[task] = mask_avg  ###保存当前的global masks
                acc_test, _, _ = test_mnist(net_glob_list[task], test_data_loader, args, prop[idx])
                acc_tasks.append(acc_test)
        else:  #######fedavg
            for task in range(1, 2):
                if iter == 1:
                    w_avg = copy.deepcopy(net_glob_list[task].state_dict())
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
                    prop[idx] = task
                    if not args.iid:  # non-iid 需要特别指定客户端的测试集    copy.deepcopy
                        local = FedAVG(args=args, dataset=data1, idxs=dict_users[idx],
                                       prop=prop[idx],
                                       checkpoints='./no_mask_{}_{}_{}_lr_{}_{}_chosen_{}_{}_eps_{}_c_{}_no_noise_{}/'.format(
                                           args.data_name, '', '', args.lr, args.num_users, m, args.dp_mechanism,
                                           args.epsilon,
                                           args.clipthr,
                                           args.noise_free), dtest=testdata, attack_type='pia',
                                idxs_test=dict_users_test[idx], update_gradient=args.EPS, poisoning=poisoning)
                    else:  # iid  
                        local = FedAVG(args=args, dataset=train_data_list[idx], idxs=None,
                                       prop=prop[idx],checkpoints='./no_mask_{}_{}_{}_lr_{}_{}_chosen_{}_{}_eps_{}_c_{}_no_noise_{}/'.format(
                                           args.data_name, '', '', args.lr, args.num_users, m, args.dp_mechanism,
                                           args.epsilon, args.clipthr, args.noise_free), dtest=testdata,
                           attack_type='pia',update_gradient=args.EPS, poisoning=poisoning)
                    w, local_acc, grad, grad_train = local.train(net=copy.deepcopy(net_glob_list[task]),
                                                                 global_epoch=iter, return_updates=True)
                    if perform_attack:
                        dlg_attack(net_glob_list[task], args, grad_train, local.ldr_train.dataset, iter, idx,
                                   w, w, fedavg=True, dm=dm, ds=ds,
                                   save_path='fedavg_{}_iid_{}_{}_{}_{}_{}_EPS_{}_ldp_{}_poison_{}'.format(args.no_mask,
                                                                                                           args.iid,
                                                                                                           args.model,
                                                                                                           args.data_name,
                                                                                                           target,
                                                                                                           privacy,
                                                                                                           args.EPS,
                                                                                                           epsilon,
                                                                                                           args.poison))
                    w_locals.append(w)  ##model updates
                    local_test_acc.append(local_acc)
                if args.poison > 0:
                    similarity = compute_similarity_poison_fedavg(w_avg, w_locals,
                        poison_label)  ###similarity = {'malicious': [],'benign':[]}

                w_avg = FedAvg(w_locals, w_global=w_avg, return_updates=True)  ####,poison_label=poison_labelaggregation
                net_glob_list[task].load_state_dict(w_avg)
                acc_test, _, _ = test_mnist(net_glob_list[task], test_data_loader, args, prop[idx])
                acc_tasks.append(acc_test)

        with open('fedavg_{}_iid_{}_{}_{}_{}_{}_EPS_{}_ldp_{}_poison_{}_{}.txt'.format(args.no_mask, args.iid, args.model,
                                                                                    args.data_name, target, privacy,
                                                                                    args.EPS, epsilon, args.poison,args.defense),
                  'a+') as f:
            f.write('Round\t{}\tAcc\t{}\tLocal Acc\t{}\n'.format(iter, acc_tasks[0], np.mean(local_test_acc)))
            print('Round\t{}\tAcc\t{}\tLocal Acc\t{}\n'.format(iter, acc_tasks[0], np.mean(local_test_acc)))
        if args.poison > 0:
            with open('cos_fedavg_{}_iid_{}_{}_{}_{}_{}_EPS_{}_ldp_{}_poison_{}_{}.txt'.format(args.no_mask, args.iid,
                                                                                            args.model, args.data_name,
                                                                                            target, privacy, args.EPS,
                                                                                            epsilon, args.poison,args.defense),
                      'a+') as f:
                f.write(f'Round\t{iter}\tMalicious\t')
                f.write(str(np.mean(similarity['malicious'], axis=0))[1:-1] + '\n')
                f.write(f'Round\t{iter}\tBenign\t')
                f.write(str(np.mean(similarity['benign'], axis=0))[1:-1] + '\n')#
