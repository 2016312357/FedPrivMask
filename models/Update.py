#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import os
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import random_split
from Calculate import get_1_norm, get_2_norm
from MIA import mia
from Noise_add import clipping
from infocom.perturb import slicing
from models.test import test_mnist, test_img
from models.gc import binary_top_k, sparse_top_k, sparse_quantile_k
import time
import torchvision

# from rog_attack.rog_attack import run_attack
from sensitivity import compute_sens_all_layer


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


class LocalUpdate(object):

    def __init__(self, args, dataset=None, idxs=None, prop=None, checkpoints='ckp', dtest=None, attack_type='mia',
                 idxs_test=None,
                 update_gradient=0, poisoning=False):
        '''
        初始化本地训练过程
        :param args: 所有参数
        :param dataset: 数据集
        :param idxs: 列表，client id of selected_clients
        :param prop: 敏感属性值，分别对应 idxs的客户端
        :param checkpoints: 保存路径名
        :param dtest: 测试集
        :param attack_type: 攻击类型:'pia'属性攻击，'mia'成员攻击
        '''
        self.poisoning = poisoning
        self.net = None  # global model
        self.args = args  # parsed hyper-parameters
        self.client_id = idxs  # client id of selected_clients
        self.loss_func = nn.CrossEntropyLoss()  # 损失函数
        self.prop = prop  # selected_clients的敏感属性
        self.attack_type = attack_type  # 攻击类型：成员推理、属性推理
        self.update_gradient = update_gradient

        if idxs is not None:  # noniid
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
            # d_train=self.ldr_train.dataset###dataloader
            self.trainset = self.ldr_train.dataset
            if dtest is not None:  # 按照参数传递中规定的数据集
                self.testset = dtest
                self.ldr_test = DataLoader(self.testset, batch_size=self.args.bs, shuffle=False)
            else:
                self.ldr_test = DataLoader(DatasetSplit(dtest, idxs_test), batch_size=self.args.bs, shuffle=False)
                self.testset = self.ldr_test.dataset
        else:
            d = dataset
            # 设置训练集、测试集
            if dtest is not None:  # 按照传递参数中规定的数据集
                self.trainset = d
                self.testset = dtest
            else:  # 随机划分训练集测试集
                self.trainset, self.testset = random_split(d, [int(len(d) * 0.5), len(d) - int(len(d) * 0.5)])
            self.ldr_train = DataLoader(self.trainset, batch_size=self.args.local_bs, shuffle=True)
            self.ldr_test = DataLoader(self.testset, batch_size=self.args.bs, shuffle=False)

        print('local train on', len(self.ldr_train.dataset), 'test on ', len(self.ldr_test.dataset), 'learning rate ',
              self.args.lr)

        # 计算DP噪声的sigma
        if args.dp_mechanism == 'laplace':
            self.sensitivity = 2. * args.clipthr / len(self.ldr_train.dataset)  # 计算敏感度S=2*C/N
            self.sigma = np.sqrt(2) * self.sensitivity / args.epsilon
        elif args.dp_mechanism == 'gauss':
            #we utilize the method in [29] and choose C by taking the median of the norms of the unclipped parameters over the course of training.
            self.sensitivity = 2. * args.clipthr / len(self.ldr_train.dataset)  # 计算敏感度S=2*C/N
            self.sigma = np.sqrt(2 * np.log(1.25 / args.delta)) * self.sensitivity / args.epsilon
            print(args.dp_mechanism, 'sigma',self.sigma, 'epsilon',args.epsilon, 'sensitivity',self.sensitivity)
        else:
            self.sigma = None
        if self.args.verbose:
            if not os.path.isdir(checkpoints):
                os.makedirs(checkpoints)
        self.checkpoints = checkpoints

    def train(self, net, global_epoch, return_updates=True):
        '''
        function:FL一轮中客户端本地进行模型更新
        input:global model
        output:返回本轮更新后的模型参数，模型生成的梯度用于训练攻击模型
        '''

        self.net = net
        init_test_acc = self.test(self.ldr_test)[0]
        print('Initial local testing acc ', init_test_acc)
        net_global = copy.deepcopy(net)

        if self.args.defense == 'compensate':
            flag = True
            sensitivity = compute_sens_all_layer(model=net,
                                             rootset_loader=self.ldr_test, device=self.args.device)
            a=iter(sensitivity)

        layer_name = []
        for k, param in net.named_parameters():  ##### state_dict().keys():###只用weight参数做攻击
            if 'weight' in k:
                layer_name.append(k)
                if self.args.defense == 'compensate':
                    print('weight sensitive',next(a))
            else:
                if self.args.defense == 'compensate':
                    _ = next(a)

        print(net_global.state_dict().keys(), layer_name)
        grad_train_list = {k: [] for k in layer_name}  # 保存训练过程中产生的梯度
        lr = self.args.lr
        # 根据攻击类型设置优化器和学习率
        if self.args.model=='cnn':
            optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=0.9, weight_decay=2e-5)  # 0.00005
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)  # 0.00005
        epoch_loss = []
        # train and update local model
        for ep in range(self.args.local_ep):
            # start = time.time()
            batch_loss = []
            correct = 0.
            if self.args.train_bn:
                net.train()
            else:
                print('freeze BN layer')
                net.train()
                for module in net.modules():# Set the BNs to eval mode so that the running means and averages do not update.
                    if 'BatchNorm' in str(type(module)):
                        module.eval()
            # 固定此时的初始模型，便于计算模型的参数更新
            if self.attack_type == 'pia':
                w_before = copy.deepcopy(net.state_dict())
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # if self.args.data_name == 'mnist':
                #     if self.prop == 0:
                #         labels = labels % 2
                #     elif self.prop == 2:
                #         labels[labels < 5] = 0
                #         labels[labels != 0] = 1
                # elif self.args.data_name == 'MotionSense' and self.args.privacy == '':
                #     labels = labels[:, self.prop]
                # elif (self.args.data_name == 'CelebA' or self.args.data_name == 'lfw') and self.args.privacy == '':
                #     labels = labels[self.prop]
                # else:
                #     pass
                if self.poisoning:
                    print('poisoning attack label flipping')
                    labels = self.args.num_classes - labels - 1
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                if self.args.defense == 'soteria':
                    net.zero_grad()
                    images.requires_grad = True
                log_probs = net(images)
                if self.args.optim == 'fedprox' and ep>0 and batch_idx>0:  # 判断FL算法类型
                    prox_term = torch.tensor(0., device=self.args.device)  # fedprox特有的正则化系数，可以减少noniid的影响
                    for w, w_t in zip(net.parameters(), net_global.parameters()):
                        # update the proximal term  # prox_term += (w - w_t).norm(2)
                        prox_term += torch.pow(torch.norm(w - w_t), 2)
                    loss = self.loss_func(log_probs, labels) + prox_term * self.args.mu/2
                else:  # fedavg
                    loss = self.loss_func(log_probs, labels)
                batch_loss.append(loss.item())
                if self.args.defense == 'outpost':
                    loss.backward()
                    input_parameters = []
                    sensitivity = []
                    for k,param in net.named_parameters():
                        sensitivity.append(torch.var(param.data).cpu().detach().numpy())
                        input_parameters.append(param.grad)
                        if 'weight' in k:
                            print(k,'layer risk outpost',sensitivity[-1])
                    sensitivity = [min(v, 1)/sum(sensitivity) for v in sensitivity]  ###layer risk
                    phi, prune_base, noise_base = self.args.epsilon[0], 90, self.args.epsilon[1]#######1e3
                    from outpost import noise
                    input_parameters = noise(dy_dx=input_parameters, phi=phi, prune_base=prune_base,
                                             noise_base_value=noise_base,risk=sensitivity)
                    param_gen = iter(input_parameters)
                    for param in net.parameters():
                        param.grad = next(param_gen).to(self.args.device)
                # elif self.args.defense == 'compensate':
                #
                #     loss.backward()
                #     input_parameters = []
                #     for name,param in net.named_parameters():
                #         if param.grad is not None:
                #             input_parameters.append(param.grad)
                #         else:
                #             print(name,'hi')
                #     slices_num = 10
                #     scale = self.args.epsilon
                #     perturb_slices_num = 7
                #     # print(len(input_parameters), len(sensitivity))
                #     # assert len(input_parameters) == len(sensitivity)
                #     from infocom.perturb import noise
                #     # Slicing gradients and random perturbing all layers, including weights and bias
                #     if flag:
                #         layer_dims_pool, layer_params_num_pool, layer_params_num_gcd, slice_indices, slice_params_indice = slicing(
                #         input_parameters, sensitivity, slices_num)
                #         flag = False
                #     input_parameters = noise(layer_dims_pool, layer_params_num_pool, layer_params_num_gcd, slice_indices, slice_params_indice,dy_dx=input_parameters,
                #                             sensitivity=sensitivity,
                #                             slices_num=slices_num,
                #                             perturb_slices_num=perturb_slices_num,
                #                             scale=scale)
                #     param_gen = iter(input_parameters)
                #     for param in net.parameters():
                #         param.grad = next(param_gen).to(self.args.device)

                elif self.args.defense == 'soteria':
                    fc_pruning_rate = self.args.epsilon
                    feature_fc1_graph = net.extract_feature()
                    deviation_f1_target = torch.zeros_like(feature_fc1_graph)
                    deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
                    for f in range(deviation_f1_x_norm.size(1)):
                        deviation_f1_target[:, f] = 1
                        feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)  ###逐层反向传播
                        deviation_f1_x = images.grad.data

                        deviation_f1_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1),
                                                               dim=1) / (
                                                            feature_fc1_graph.data[:, f] + 0.1)
                        net.zero_grad()
                        images.grad.data.zero_()
                        deviation_f1_target[:, f] = 0
                    deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)  # [512],fc input length
                    thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), fc_pruning_rate)
                    mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
                    # input_gradient = torch.autograd.grad(target_loss, model.parameters())
                    loss.backward()
                    input_gradient = [param.grad for param in net.parameters()]
                    input_gradient[-2] = input_gradient[-2] * torch.Tensor(mask).to(self.args.device)
                else:
                    loss.backward()
                # if self.update_gradient <= 1:  # self.args.mask_scale_gradients == 'top':
                #     for name, p in net.named_parameters():
                #         if p.grad is not None:
                #             # p.grad = sparse_quantile_k(p.grad, input_compress_settings={'k': self.update_gradient,
                #             #          'k1': self.update_gradient-0.1},dp_noise=self.sigma)
                #
                #             p.grad.data = sparse_top_k(p.grad.data,
                #                                        input_compress_settings={'k': self.update_gradient})
                optimizer.step()  # update local model
                log_probs = log_probs.float().cpu().data.numpy()
                labels = labels.float().cpu().data.numpy()
                correct += np.sum(labels == np.argmax(log_probs[:], axis=1))

                # if self.attack_type == 'pia_grad':  # 属性攻击，用每个batch内参数更新（梯度）来训练attack model
                #     w_new = copy.deepcopy(net.state_dict())
                #     for k in grad_train_list.keys():
                #         tmp = (w_new[k] - w_before[k]) * 1.0 / self.args.lr
                #         if 'weight' in k and self.args.gc < 1:
                #             tmp = binary_top_k(tmp, input_compress_settings={'k': self.args.gc})  # sparsity
                #         grad_train_list[k].extend(
                #             tmp.cpu().numpy().reshape((1, -1)))  # (batch_idx+1,[[],[]], /

            accuracy = correct / len(self.ldr_train.dataset)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if ep % 10 == 0 or (ep + 1) == self.args.local_ep:
                # interval = time.time() - start'train one epoch past time', interval,
                print(self.args.defense, self.args.optim,'Epoch:{} Average training loss: {} \tAccuracy: {}/{} ({}) '.format(
                    ep, epoch_loss[-1], correct, len(self.ldr_train.dataset), accuracy))
            if self.attack_type == 'pia':  # 属性攻击，为了减少训练数据量，还可以用每个epoch内的模型更新（梯度）来train attack
                w_new = net.state_dict()
                for k in grad_train_list.keys():
                    tmp = (w_new[k] - w_before[k]) * 1.0 / self.args.lr / len(self.ldr_train)
                    # if self.update_gradient < 1:
                    #     tmp = sparse_top_k(tmp, input_compress_settings={'k': self.update_gradient})
                    grad_train_list[k].extend(tmp.cpu().numpy().reshape((1, -1)))  # (batch_idx+1,[[],[]], /
        if self.args.defense == 'compensate':
            input_parameters = []
            for name, param in net.named_parameters():
                input_parameters.append(param.data.detach())
            slices_num = 10
            scale = self.args.epsilon
            perturb_slices_num = 7
            from infocom.perturb import noise
            if flag:
                layer_dims_pool, layer_params_num_pool, layer_params_num_gcd, slice_indices, slice_params_indice = slicing(
                    input_parameters, sensitivity, slices_num)
                flag = False
            input_parameters = noise(layer_dims_pool, layer_params_num_pool, layer_params_num_gcd, slice_indices,
                                     slice_params_indice, dy_dx=input_parameters,
                                     sensitivity=sensitivity,
                                     slices_num=slices_num,
                                     perturb_slices_num=perturb_slices_num,
                                     scale=scale)

            param_gen = iter(input_parameters)
            for param in net.parameters():
                param.data = next(param_gen).to(self.args.device)

        w = copy.deepcopy(net.state_dict())
        # if self.sigma is not None and global_epoch not in self.args.noise_free:
        #     print('adding DP')
        #     w = clipping(self.args, w)  # 裁剪限定参数大小
        #     for k in w.keys():
        #         if 'weight' in k or 'bias' in k:
        #             # print('adding noise to', k, self.sigma)
        #             if self.args.dp_mechanism == 'gauss':
        #                 noise = np.random.normal(0, self.sigma, w[k].size())
        #             elif self.args.dp_mechanism == 'laplace':
        #                 noise = np.random.laplace(0, self.sigma, w[k].size())
        #             noise = torch.from_numpy(noise).float().to(self.args.device)
        #             w[k] += noise
        #     self.net.load_state_dict(w)  # 加载加噪后的模型
        #     # print('Training acc after adding dp noise', self.test(self.ldr_train))
        #     print('Testing acc after adding LDP defense', self.test(self.ldr_test))

        # 属性推理攻击，保存模型整体的参数更新作为攻击模型的测试数据
        w_g = net_global.state_dict()  ### init model
        grad_list = {k: [] for k in layer_name}
        # start = time.time()
        if self.sigma is not None and global_epoch not in self.args.noise_free:
            w = clipping(self.args, w)  # clipping参数L2 norm
        for k in w_g.keys():###bn layers also aggregate or 'bias' in k
            w_updates = w[k] - w_g[k]
            if self.update_gradient < 1:
                # print('pruning DP noise')
                # w_updates = sparse_top_k(w_updates,
                #   input_compress_settings={'k': self.update_gradient},dp_noise=self.sigma)
                # w_updates = sparse_quantile_k(w_updates, input_compress_settings={'k': self.update_gradient,
                #             'k1': self.update_gradient-0.1},dp_noise=self.sigma)
                noise_mask = binary_top_k(w_updates, input_compress_settings={'k': self.update_gradient})
                noise = np.random.laplace(0, self.sigma, w[k].size())

                noise = torch.from_numpy(noise).float().to(self.args.device)
                noise=noise * noise_mask
                print('masked noise',(noise==0).sum()/noise.numel())
                w_updates += noise
                tmp = w_updates * 1.0 / self.args.local_ep / self.args.lr / len(self.ldr_train)
            elif self.sigma is not None and global_epoch not in self.args.noise_free:
                print('adding DP to all params',self.sigma)
                if self.args.dp_mechanism == 'gauss':
                    noise = np.random.normal(0, self.sigma, w[k].size())
                elif self.args.dp_mechanism == 'laplace':
                    noise = np.random.laplace(0, self.sigma, w[k].size())
                noise = torch.from_numpy(noise).float().to(self.args.device)
                w_updates += noise
                tmp = w_updates * 1.0 / self.args.local_ep / self.args.lr / len(self.ldr_train)
            else:
                tmp = w_updates * 1.0 / self.args.local_ep / self.args.lr / len(self.ldr_train)
            if k in grad_list.keys():
                grad_list[k].extend(tmp.cpu().numpy().reshape((1, -1)))
                grad_list[k] = np.vstack(grad_list[k])
                if self.attack_type == 'pia':
                    grad_train_list[k] = np.vstack(grad_train_list[k])
            if return_updates:
                w[k] = w_updates
        return w, init_test_acc, grad_list, grad_train_list  # {}sum(epoch_loss)/len(epoch_loss),param , test_acc, test_loss, self.sigma

    def test(self, test_loader=None):
        '''
        function:测试全局模型准确率
        input:测试集
        output: 准确率和损失值
        '''
        prop = self.prop
        if test_loader is None:
            test_loader = self.ldr_test
            prop = 1 - self.prop
            print('test privacy attr', prop)
        test_acc, test_loss, targetp = test_mnist(self.net, test_loader, self.args, prop,printing=False)
        return test_acc, test_loss, targetp

    def compute_gradients(self, net, testset=None, global_net=None, prop=None):
        '''
        function:成员推理攻击需要保存模型输出的概率向量作为攻击模型的训练数据
        :param net: 更新后的全局模型
        :param testset: 测试集
        :param global_net: 旧的全局模型
        :return: 模型输出的概率向量的变化量
        '''
        test_loader = DataLoader(testset, batch_size=64, shuffle=False)
        # output = None
        output = []
        label_list = []  ##priv
        label_list2 = []  ##target
        with torch.no_grad():
            net.eval()
            # global_net.eval()
            for sample_idx, (images, labels) in enumerate(test_loader):
                # if sample_idx >= 5:  # 为了减少内存消耗，只保留少部分数据的概率输出
                #     break
                labels2 = labels[1 - prop]  # target
                labels = labels[prop]  # privacy
                images, labels = images.to(self.args.device), labels.to(self.args.device)  # .unsqueeze(0)

                # log_probs = net(images)  # logit output

                new_m = torchvision.models._utils.IntermediateLayerGetter(net,
                                                                          {'shared': 'feat1'})
                new_m_out = new_m(images)
                for k, v in new_m_out.items():
                    output.append(v.view(v.size(0), -1).cpu())
                label_list.append(labels.cpu())
                label_list2.append(labels2.cpu())

            output = np.vstack(output)
            label_list = np.concatenate(label_list)
            label_list2 = np.concatenate(label_list2)

        return output, label_list, label_list2

    def adv_train(self, net, global_epoch, target):
        '''
        function:FL一轮中客户端本地进行模型更新
        input:global model
        output:返回本轮更新后的模型参数，模型生成的梯度用于训练攻击模型
        '''
        self.net = net
        init_test_acc = self.test(self.ldr_test)
        print('Initial Testing (acc,loss,fairness) ', init_test_acc)
        net_global = copy.deepcopy(net)  ###initial net
        if self.attack_type == 'mia':
            # 成员推理攻击只需要保存模型输出的概率向量作为攻击模型的训练数据
            mem, mem_label, mem_label_target = self.compute_gradients(net_global, self.trainset, net_global,
                                                                      1 - self.prop)  # 训练集：member
            non_mem, nom_mem_label, non_mem_label_target = self.compute_gradients(net_global, self.testset, net_global,
                                                                                  1 - self.prop)  # 测试集：非成员
            init_acc_priv, init_auc_priv = mia(mem, mem_label, non_mem, nom_mem_label,
                                               self.args, iter=0)
            init_acc_uti, init_auc_uti = mia(None, None, non_mem, non_mem_label_target,
                                             self.args, iter=0)
        layer_name = []
        for k in net.state_dict().keys():
            if 'weight' in k:
                layer_name.append(k)
                # grad_train_list = {k: [] for k in net_global.state_dict().keys()}  # 保存训练过程中产生的梯度
        E = net.shared
        C = net.classifier
        E_optimizer = torch.optim.Adam(
            net.shared.parameters(),
            lr=self.args.lr)  ##self.args.lr_mask feature extractor layers,不同的dataset classifier任务对应不同的mask
        C_optimizer = torch.optim.Adam(
            net.classifier.parameters(), lr=self.args.lr)
        P = copy.deepcopy(net.classifier)
        P_optimizer = torch.optim.Adam(
            P.parameters(), lr=self.args.lr)
        print('adv target model learning rate', self.args.lr, "target prop:", self.prop)
        # train and update local model
        for iter in range(self.args.local_ep):
            Eloss = []
            Ploss = []
            Closs = []
            net.train()
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # if self.attack_type == 'pia_grad':
                #     w_before = copy.deepcopy(net.state_dict())
                if self.args.data_name == 'mnist':
                    if self.prop == 0:
                        labels = labels % 2
                    elif self.prop == 2:
                        labels[labels < 5] = 0
                        labels[labels != 0] = 1
                elif self.args.data_name == 'MotionSense' and self.args.privacy == '':
                    labels = labels[:, self.prop]
                elif (self.args.data_name == 'CelebA' or self.args.data_name == 'lfw') and self.args.privacy == '':
                    pri_labels = labels[1 - self.prop]
                    labels = labels[self.prop]

                if self.poisoning:
                    print('poisoning attack label flipping')
                    labels = self.args.num_classes - labels - 1
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                E.train()
                C.train()
                P.eval()
                task1 = labels.to(self.args.device)
                task2 = pri_labels.to(self.args.device)
                E_optimizer.zero_grad()
                C_optimizer.zero_grad()
                features = E(images)
                features = features.view(features.size(0), -1)

                out1 = C(features)  # .float()  # log_softmax
                out2 = P(features)  # .float()
                C_loss = self.loss_func(out1, task1.long())  # loss(out1, task1.float())
                ploss = self.loss_func(out2, task2.long())  # loss(out1, task1.float())

                # entropy = (nn.KLDivLoss(out2 - 0.5, exponent=2))
                # ploss = loss(out2, task2.float())
                # print(ploss.item(),torch.mean(entropy).item(),entropy)
                P_loss = -ploss  # + 0.01*torch.mean(entropy)  # -out2*torch.log2(out2)# (torch.mean(torch.norm(out2 - 0.5)))
                E_loss = (1 - self.args.tradeoff_lambda) * C_loss + self.args.tradeoff_lambda * P_loss
                E_loss.backward()
                C_optimizer.step()
                E_optimizer.step()
                Eloss.append(E_loss.item())
                Closs.append(C_loss.item())
                if self.args.tradeoff_lambda > 0:
                    E.eval()
                    with torch.no_grad():
                        features = E(images)
                        features = features.view(features.size(0), -1)
                    for _ in range(3):
                        P_optimizer.zero_grad()
                        out2 = P(features.detach())  # .float()
                        P_loss = self.loss_func(out2, task2.long())  # loss(out2, task2.float())
                        P_loss.backward()
                        P_optimizer.step()
                        Ploss.append(P_loss.item())

            print(iter, 'adv acc for target:{}, loss for E:{}, C:{}, P:{}\n'.format(self.test(self.ldr_test)[0],
                                                                                    np.mean(Eloss), np.mean(Closs),
                                                                                    np.mean(Ploss)))
            if self.args.tradeoff_lambda == 0 and np.mean(Eloss) <= 0.001:
                break
        w = net.state_dict()
        # if not os.path.exists(self.checkpoints):
        #     os.makedirs(self.checkpoints)
        # torch.save(net.state_dict(), self.checkpoints+target[0]+target[1]+".pth")
        print('Final Training acc after adding noise', self.test(self.ldr_train), target[self.prop])
        print('Final Testing acc after adding noise', self.test(self.ldr_test), target[self.prop])
        if self.attack_type == 'mia':
            # 成员推理攻击只需要保存模型输出的概率向量作为攻击模型的训练数据
            mem, mem_label, mem_label_target = self.compute_gradients(self.net, self.trainset, net_global,
                                                                      1 - self.prop)  # 训练集：member
            non_mem, nom_mem_label, non_mem_label_target = self.compute_gradients(self.net, self.testset, net_global,
                                                                                  1 - self.prop)  # 测试集：非成员
            acc_priv, auc_priv, init_acc_priv1, init_auc_priv1 = mia(mem, mem_label, non_mem, nom_mem_label, self.args,
                                                                     iter=self.args.mia_ep)
            acc_utility, auc_utility, init_acc_uti1, init_auc_uti1 = mia(None, None, non_mem,
                                                                         non_mem_label_target, self.args,
                                                                         iter=self.args.mia_ep)  # mem, mem_label_target
            print('lambda\t', self.args.tradeoff_lambda, 'utility:', target[self.prop],
                  acc_utility, auc_utility, 'privacy:', target[1 - self.prop], acc_priv, auc_priv)
            with open('./{}_{}.txt'.format(self.args.model, self.args.data_name), 'a+') as f:
                f.write('{}-{}\tlambda\t{}\tUtility\t{}\t{}\t{}\t{}\tPrivacy\t{}\t{}\t{}\t{}\n'.format(
                    target[self.prop], target[1 - self.prop], self.args.tradeoff_lambda,
                    init_acc_uti, init_auc_uti, acc_utility, auc_utility,
                    init_acc_priv, init_auc_priv, acc_priv, auc_priv))
            return w, init_test_acc, mem, mem_label, non_mem, nom_mem_label  # , test_acc, test_loss, self.sigma
