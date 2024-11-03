"""Main entry point for doing all stuff."""
from __future__ import division, print_function

import copy
import os
import time

import torch
import torch.nn as nn
# import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader
# from tqdm import tqdm
import numpy as np
# from models.dp import alg1
from models.gc import sparse_top_k, sparse_top_k_index
import networks as net
from sensitivity import compute_sens
import utils.utils as utils
from models.test import test_mnist
import math


# torch.random.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
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
    def __init__(self, args, model, dataset=None, idxs=None, prop=None, checkpoints='ckp', dtest=None,
                 attack_type='mia', idxs_test=None, update_gradient=0, poisoning=False):
        '''
        初始化本地训练过程
        :param args: 所有参数
        :param dataset: 数据集
        :param idxs: data sample id for non-iid distribution
        :param prop: 敏感属性值，分别对应 idxs的客户端
        :param checkpoints: 保存路径名
        :param dtest: 测试集
        :param attack_type: 攻击类型:'pia'属性攻击，'mia'成员攻击
        '''

        self.poisoning = poisoning
        self.update_gradient = update_gradient
        self.net = None  # global model
        self.args = args  # parsed hyper-parameters
        # self.client_id = idxs  # client id of selected_clients
        # self.loss_func = nn.CrossEntropyLoss()  # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.prop = prop  # selected_clients的敏感属性
        self.attack_type = attack_type  # 攻击类型：mia成员推理、pia属性推理
        if idxs is not None:  # noniid
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
            if dtest is not None:  # 按照参数传递中规定的数据集
                self.testset = dtest
                self.ldr_test = DataLoader(self.testset, batch_size=self.args.bs, shuffle=False)
            else:
                self.ldr_test = DataLoader(DatasetSplit(dtest, idxs_test), batch_size=self.args.bs, shuffle=False)

        else:
            d = dataset
            # 设置训练集、测试集
            if dtest is not None:  # 按照参数传递中规定的数据集
                d_train = d
                self.testset = dtest
            else:  # 随机划分训练集测试集
                d_train, d_test = random_split(d, [int(len(d) * 0.8), len(d) - int(len(d) * 0.8)])
                self.testset = d_test
            self.trainset = d_train
            self.ldr_train = DataLoader(self.trainset, batch_size=self.args.local_bs, shuffle=True)
            self.ldr_test = DataLoader(self.testset, batch_size=self.args.bs, shuffle=False)

        print('local train on', len(self.ldr_train.dataset), 'test on ', len(self.ldr_test.dataset))

        if self.args.verbose:
            if not os.path.isdir(checkpoints):
                os.makedirs(checkpoints)
        self.checkpoints = checkpoints
        self.cuda = torch.cuda.is_available()
        self.model = model
        self.net_global=copy.deepcopy(model)
        self.optimizer = Optimizers(self.args)

        if self.args.model == 'cnn':
            # optimizer_shared = torch.optim.Adam(
            #     model.shared.parameters(), lr=args.lr_mask,
            #     weight_decay=0.0001)
            optimizer_masks = torch.optim.Adam(
                model.parameters(), lr=args.lr_mask,
                weight_decay=2e-4)  # feature extractor layers,不同的dataset classifier任务对应不同的mask
        else:
            optimizer_masks = torch.optim.RMSprop(
                model.parameters(), lr=args.lr_mask,
                weight_decay=0.0002)  # )# feature extractor layers,不同的dataset classifier任务对应不同的mask
        # self.optimizer.add(optimizer_shared, self.args.lr_mask,
        #                    self.args.lr_mask_decay_every)
        self.optimizer.add(optimizer_masks, self.args.lr_mask,
                           self.args.lr_mask_decay_every)

        # optimizer_masks = optim.Adam(
        #     self.model.shared.parameters(),
        #     lr=self.args.lr_mask)  ##self.args.lr_mask feature extractor layers,不同的dataset classifier任务对应不同的mask
        # optimizer_classifier = optim.Adam(
        #     self.model.classifier.parameters(), lr=self.args.lr_classifier)#self.args.lr_classifier
        # # state_dict()
        # self.optimizer = Optimizers(self.args)
        # self.optimizer.add(optimizer_masks, self.args.lr_mask,
        #                    self.args.lr_mask_decay_every)
        # self.optimizer.add(optimizer_classifier, self.args.lr_classifier,
        #                    self.args.lr_classifier_decay_every)
        # for name, p in model.named_parameters():
        #     print(name, p.requires_grad,)#p.data

    def eval(self):
        tr_acc, tr_loss, _ = test_mnist(self.model, self.ldr_train, self.args, self.prop)
        print(self.args.data_name, self.prop, 'task, local client train acc:', tr_acc, 'loss:', tr_loss)

        test_acc, test_loss, _ = test_mnist(self.model, self.ldr_test, self.args, self.prop)
        print(self.prop, 'task, local client test acc:', test_acc, 'loss:', test_loss)
        if self.args.train_bn:
            self.model.train()
        else:
            self.model.train_nobn()

        return test_acc, test_loss

    def do_batch(self, optimizer, batch, label, norm,epoch_idx,batch_idx):
        """Runs model for one batch."""
        if self.args.train_bn:
            self.model.train()
        else:  # freeze batchnorm layer
            # print('freeze bn layer')
            self.model.train_nobn()
        if self.args.data_name == 'mnist':
            if self.prop == 0:
                label = label % 2
            elif self.prop == 2:
                label[label < 5] = 0
                label[label != 0] = 1
        elif self.args.data_name == 'MotionSense' and self.args.privacy == '':
            label = label[:, self.prop]
            print('motion sense task training label', label)
        elif isinstance(label, list) and (
                self.args.data_name == 'CelebA' or self.args.data_name == 'lfw') and self.args.privacy == '':
            pri_labels = label[1 - self.prop]
            label = label[self.prop]
        elif (self.args.data_name == 'CelebA' or self.args.data_name == 'lfw') and self.args.privacy == '':
            label = label
            # label = label[self.prop]

        if self.poisoning:
            # print('poisoning attack label flipping before',label[:2])
            label = self.args.num_classes - label - 1
            # print('poisoning attack label flipping after',label[:2])

        if self.cuda:
            batch = batch.to(self.args.device)
            label = label.to(self.args.device)
        batch = Variable(batch)
        label = Variable(label)
        # Set grads to 0.
        self.model.zero_grad()
        self.optimizer.zero_grad()
        # Do forward-backward.
        # start=time.time()
        output = self.model(batch)

        prox_term = 0.  # fedprox特有的正则化系数，可以减少noniid的影响
        if self.args.optim == 'fedprox' and self.args.mu>0:  # 判断FL算法类型
            if batch_idx>0 and epoch_idx>0:
                for w, w_t in zip(self.model.parameters(), self.net_global.parameters()):
                    # update the proximal term
                    prox_term += torch.pow(torch.norm(w - w_t), 2)
                # loss = self.loss_func(log_probs, labels) + prox_term * (self.args.mu / 2)

        tr_loss = self.criterion(output, label) + self.args.mu * prox_term###2e-4.float().float()

        tr_loss.backward()
        # interval=time.time()-start
        # print('one batch using',interval)
        # for module_idx, module in enumerate(self.model.modules()):  # importance of each grad
        #     if 'ElementWise' in str(type(module)):
        #         if module_idx in norm.keys():
        #             norm[module_idx] += module.mask_real.grad.data
        #         else:
        #             norm[module_idx] = module.mask_real.grad.data

        # Scale gradients by average weight magnitude.
        if self.args.mask_scale_gradients != 'none':
            # print('scaling gradients')
            for module in self.model.modules():  # shared.
                if 'ElementWise' in str(type(module)):
                    abs_weights = module.weight.data.abs()
                    if self.args.mask_scale_gradients == 'average':
                        module.mask_real.grad.data.div_(abs_weights.mean())
                    elif self.args.mask_scale_gradients == 'individual':
                        module.mask_real.grad.data.div_(abs_weights)

        if not self.args.train_bn:  # Set bn layer grads to 0, if required.
            for module in self.model.modules():  # shared.
                if 'BatchNorm' in str(type(module)):
                    if module.weight.grad is not None:
                        module.weight.grad.data.fill_(0)
                    if module.bias.grad is not None:
                        module.bias.grad.data.fill_(0)
                    if module.running_mean is not None:
                        module.running_mean.grad.data.fill_(0)
                    if module.running_var is not None:
                        module.running_var.grad.data.fill_(0)
        # Update params.
        self.optimizer.step()
        return tr_loss.item()

    def do_epoch(self, epoch_idx, optimizer, norm):
        """Trains model for one epoch."""
        ep_loss = 0.

        for batch_idx, (batch, label) in enumerate(self.ldr_train):  # , desc='Epoch: %d ' % (epoch_idx)):
            ep_loss += self.do_batch(optimizer, batch, label, norm,epoch_idx,batch_idx)
        ep_loss /= len(self.ldr_train)  # batch内loss使用的参数为mean, 会对N个样本的loss进行平均之后返回
        return ep_loss
        # if self.args.threshold_fn == 'binarizer':
        #     # print('Num 0ed out parameters:')
        #     for idx, module in enumerate(self.model.shared.modules()):
        #         if 'ElementWise' in str(type(module)):
        #             num_zero = module.mask_real.data.lt(5e-3).sum()  # mask各位置值是否小于5e-3
        #             total = module.mask_real.data.numel()
        #             print('binary',idx, num_zero, total)
        # elif self.args.threshold_fn == 'ternarizer':
        #     # print('Num -1, 0ed out parameters:')
        #     for idx, module in enumerate(self.model.shared.modules()):
        #         if 'ElementWise' in str(type(module)):
        #             num_neg = module.mask_real.data.lt(0).sum()
        #             num_zero = module.mask_real.data.lt(5e-3).sum() - num_neg
        #             total = module.mask_real.data.numel()
        #             print('ternarize, -1,0, total', idx, num_neg, num_zero, total)

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        # Prepare the ckpt.
        ckpt = {
            'args': self.args,
            # 'epoch': epoch,
            # 'accuracy': best_accuracy,
            # 'errors': errors,
            # 'model': self.model.state_dict(),
            'mask': self.model.named_parameters(),
        }
        # Save to file.
        torch.save(ckpt, savename)

    def train(self, epochs, epsilon=0, sen=None, layer_wise=True, save=False, savename='', best_accuracy=0,
              mask_ori=None,return_updates=True):
        """Performs training."""
        init_test_acc, _ = self.eval()
        print('*******above initial testing accuracy: ********')

        norm = {}  # importance of masks; torch.zeros_like(feature_fc1_graph)
        for module_idx, module in enumerate(self.model.modules()):  # compute the importance of each grad
            if 'ElementWise' in str(type(module)):
                norm[module_idx] = copy.deepcopy(module.mask_real.data)##initial real-valued mask
        for idx in range(epochs):
            start=time.time()
            epoch_idx = idx + 1
            # self.optimizer.update_lr(epoch_idx)
            if self.args.train_bn:
                self.model.train()
            else:  # freeze batchnorm layer
                self.model.train_nobn()
            loss = self.do_epoch(idx, self.optimizer, norm)

            if epoch_idx % 10 == 0:
                # print('Mask-based training for one epoch past time', time.time()-start,)
                acc, _ = self.eval()
            # if loss <= 0.001:
            #     break

        masks = {}
        masks_real = {}
        ratio = {}
        if epsilon is not None:
            p = math.exp(epsilon) / (1 + math.exp(epsilon))  # 扰动概率
            print(f'epsilon {epsilon},perturbing with', 1 - p)
        num_perturbation_layer = 0
        num_one = 0

        for module_idx, module in enumerate(self.model.modules()):  # shared.
            if 'ElementWise' in str(type(module)):
                masks_real[module_idx] = module.mask_real.data###.clone()
                mask = module.mask_real.data  # .cpu()
                norm[module_idx] -= copy.deepcopy(mask)  # total mask_real updates
                norm[module_idx] = torch.abs(norm[module_idx])
                # result = torch.sigmoid(mask)  #
                outputs = mask.clone()
                outputs.fill_(0)
                outputs[mask > 0] = 1  # save current binary mask, 0 or 1
                ######Apply LDP to mask########
                if epsilon is not None:
                    # num_perturbation_layer+=1
                    ran = torch.rand_like(outputs)
                    # print(outputs[ran > p][0],'before')
                    outputs[ran > p] = 1 - outputs[ran > p]
                    # print(outputs[ran > p][0],'flip after')
                elif self.update_gradient < 1:
                    if module_idx<=1:
                        print('pruning',1-self.update_gradient)
                    mask_updates = outputs - mask_ori[module_idx]#1,-1,0
                    mask_updates[sparse_top_k(norm[module_idx].cpu(),  # keep top-k
                         input_compress_settings={'k': self.update_gradient}) == 0] = 0 # # 不更新least 1-k的mask，保证每轮FL更新mask数目固定

                    # indices = sparse_top_k_index(norm[module_idx][mask_updates != 0].cpu(),
                    #                              input_compress_settings={
                    #                                  'k': mask_updates.numel() * (1 - self.update_gradient) -
                    #                                       norm[module_idx][mask_updates == 0].numel()})
                    # mask_updates[mask_updates != 0][indices] = 0
                    if return_updates:
                        outputs = mask_updates
                    else:
                        outputs = mask_ori[module_idx] + mask_updates
                elif self.args.k is not None:  # 按dissimilarity的比例更新
                    # mask_updates = torch.zeros_like(outputs)##mask ori未更新的部分不更新？
                    mask_updates = outputs - mask_ori[module_idx]  # 1,-1,0,
                    # norm[module_idx][mask_updates == 0] = 0  ####################3
                    neqzero = (norm[module_idx] > 0).sum().item()
                    # num_zeros = norm[module_idx].eq(0).sum().item()
                    # total = outputs.numel()
                    # assert num_zeros + neqzero == total
                    mask_updates[sparse_top_k(norm[module_idx].data.cpu(),
                                              input_compress_settings={'k': self.args.k[
                                              module_idx] * neqzero}) == 0] = 2  # pruned not to be updated
                    # outputs = mask_ori[module_idx] + mask_updates      .to(
                    #                         self.args.device)
                    outputs = mask_updates
                elif return_updates:

                    mask_updates = outputs - mask_ori[module_idx]
                    outputs = mask_updates
                else:
                    print('return mask, not updates')
                    pass
                num_one = outputs.eq(2).sum().item() ###+ outputs.eq(0).sum().item()##-1,0,1
                total = outputs.numel()
                ratio[module_idx] = (num_one+ outputs.eq(0).sum().item()) * 1.0 / total
                # ######Apply LDP########
                # if epsilon > 0 and layer_wise:
                #     print('layerwise LDP', 1 - p)
                #     if num_perturbation_layer < 2:
                #         num_perturbation_layer += 1
                #         ran = torch.rand_like(outputs)
                #         # print(outputs[ran<p])
                #         outputs[ran > p] = 1 - outputs[ran > p]  # flipping
                #         # 其正面向上的概率为p，反面向上的概率为1-p。
                #         # 若正面向上，则回答真实答案，反面向上，则回答相反的答案。
                #         # outputs[ran<p][outputs[ran<p]==1]=0
                #         # outputs[ran<p][outputs[ran<p]==0]=1
                #     else:
                #         num_perturbation_layer += 1
                #         ran = torch.rand_like(outputs)
                #         # print(outputs[ran<p])
                #         outputs[ran > (2 * p)] = 1 - outputs[ran > (2 * p)]  # flipping
                # elif epsilon > 0:
                #     print('perturbing all layers equally', 1 - p)
                #     # num_perturbation_layer+=1
                #     ran = torch.rand_like(outputs)
                #     outputs[ran > p] = 1 - outputs[ran > p]
                # if self.args.epochs == 1:
                #     print('updating local real-valued mask')
                #     module.mask_real.data[outputs == 1] = 0.001
                #     module.mask_real.data[outputs == 0] = -0.001
                if self.args.save:
                    # print('byte tensor only 1 byte of 0/1, int:4 byte', )
                    masks[module_idx] = outputs.type(torch.cuda.ByteTensor)  # 0,1 ## without -1
                else:
                    masks[module_idx] = outputs  # 0,-1,1,2
            elif self.args.train_bn and 'BatchNorm' in str(type(module)):
                masks_real[module_idx] = {}
                if module.weight is not None:
                    masks_real[module_idx]['weight'] = module.weight.data
                    masks_real[module_idx]['bias'] = module.bias.data
                if module.running_var is not None:
                    masks_real[module_idx]['running_mean'] = module.running_mean
                    masks_real[module_idx]['running_var'] = module.running_var
        # del norm
        print('unchanged mask ratio----------------------------------', ratio,num_one,'pruned')

        if save:
            if not os.path.isdir(self.checkpoints):
                os.makedirs(self.checkpoints)
            ckpt = {
                # 'args': self.args,
                # 'ones_ratio':ratio,
                'mask': masks,
                # 'norm': norm
            }
            # Save to file.
            torch.save(ckpt,
                       os.path.join(self.checkpoints, savename))  # +savename   os.path.join(self.checkpoints,savename))
            print('saving to', self.checkpoints + savename)
            # exit(0)

        return masks_real, masks, norm, init_test_acc  # sen, (self.model.classifier.state_dict()),

    def check(self):
        """Makes sure that the trained model weights match those of the pretrained model."""
        # print('Making sure filter weights have not changed.')
        if self.args.arch == 'vgg16':
            pretrained = net.ModifiedVGG16(original=True)
        elif self.args.arch == 'vgg16bn':
            pretrained = net.ModifiedVGG16BN(original=True)
        elif self.args.arch == 'resnet50':
            pretrained = net.ModifiedResNet(self.args, original=True)
        elif self.args.arch == 'densenet121':
            pretrained = net.ModifiedDenseNet(original=True)
        elif self.args.arch == 'resnet50_diff':
            pretrained = net.ResNetDiffInit(self.args.source, original=True)
        else:
            raise ValueError('Architecture %s not supported.' %
                             (self.args.arch))
        v = []
        for i in self.model.shared.modules():
            # print(str(type(i)))
            if 'Sigmoid' not in str(type(i)) and 'View' not in str(type(i)):
                # v.append(i) and 'Sigmoid' not in str(type(i)):
                v.append(i)
        # print(v)
        for module, module_pretrained in zip(v, pretrained.shared.modules()):
            print(str(type(module)), str(type(module_pretrained)))
            if 'ElementWise' in str(type(module)) or 'BatchNorm' in str(type(module)):
                weight = module.weight.data.cpu()  # weight, 不包含mask
                weight_pretrained = module_pretrained.weight.data.cpu()
                # Using small threshold of 1e-8 for any floating point inconsistencies.
                # Note that threshold per element is even smaller as the 1e-8 threshold
                # is for sum of absolute differences.
                assert (weight - weight_pretrained).abs().sum() < 1e-8, \
                    'module %s failed check' % (module)
                if module.bias is not None:
                    bias = module.bias.data.cpu()
                    bias_pretrained = module_pretrained.bias.data.cpu()
                    assert (bias - bias_pretrained).abs().sum() < 1e-8
                if 'BatchNorm' in str(type(module)):
                    rm = module.running_mean.cpu()
                    rm_pretrained = module_pretrained.running_mean.cpu()
                    assert (rm - rm_pretrained).abs().sum() < 1e-8
                    rv = module.running_var.cpu()
                    rv_pretrained = module_pretrained.running_var.cpu()
                    assert (rv - rv_pretrained).abs().sum() < 1e-8
        print('Passed checks...')


class Optimizers(object):
    """Handles a list of optimizers."""

    def __init__(self, args):
        self.optimizers = []
        self.lrs = []
        self.decay_every = []
        self.args = args

    def add(self, optimizer, lr, decay_every):
        """Adds optimizer to list."""
        self.optimizers.append(optimizer)
        self.lrs.append(lr)
        self.decay_every.append(decay_every)

    def step(self):
        """Makes all optimizers update their params."""
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        """Makes all optimizers update their params."""
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def update_lr(self, epoch_idx):
        """Update learning rate of every optimizer."""
        for optimizer, init_lr, decay_every in zip(self.optimizers, self.lrs, self.decay_every):
            optimizer = utils.step_lr(
                epoch_idx, init_lr, decay_every,
                self.args.lr_decay_factor, optimizer)

#
# def main():
#     """Do stuff."""
#     args = FLAGS.parse_args()
#
#     # Set default train and test path if not provided as input.
#     utils.set_dataset_paths(args)
#
#     # Load the required model.
#     if args.arch == 'vgg16':
#         model = net.ModifiedVGG16(mask_init=args.mask_init,
#                                   mask_scale=args.mask_scale,
#                                   threshold_fn=args.threshold_fn,
#                                   original=args.no_mask)
#     elif args.arch == 'vgg16bn':
#         model = net.ModifiedVGG16BN(mask_init=args.mask_init,
#                                     mask_scale=args.mask_scale,
#                                     threshold_fn=args.threshold_fn,
#                                     original=args.no_mask)
#     elif args.arch == 'resnet50':
#         model = net.ModifiedResNet(mask_init=args.mask_init,
#                                    mask_scale=args.mask_scale,
#                                    threshold_fn=args.threshold_fn,
#                                    original=args.no_mask)
#     elif args.arch == 'densenet121':
#         model = net.ModifiedDenseNet(mask_init=args.mask_init,
#                                      mask_scale=args.mask_scale,
#                                      threshold_fn=args.threshold_fn,
#                                      original=args.no_mask)
#     elif args.arch == 'resnet50_diff':
#         assert args.source
#         model = net.ResNetDiffInit(args.source,
#                                    mask_init=args.mask_init,
#                                    mask_scale=args.mask_scale,
#                                    threshold_fn=args.threshold_fn,
#                                    original=args.no_mask)
#     else:
#         raise ValueError('Architecture %s not supported.' % (args.arch))
#
#     # Add and set the model dataset.
#     model.add_dataset(args.dataset, args.num_outputs)
#     model.set_dataset(args.dataset)#定义self.classifier = self.classifiers[self.datasets.index(dataset)]
#
#     if args.cuda:
#         model = model.cuda()
#
#     # Initialize with weight based method, if necessary.
#     if not args.no_mask and args.mask_init == 'weight_based_1s':
#         print('Are you sure you want to try this?')
#         assert args.mask_scale_gradients == 'none'
#         assert not args.mask_scale
#         for idx, module in enumerate(model.shared.modules()):
#             if 'ElementWise' in str(type(module)):
#                 weight_scale = module.weight.data.abs().mean()
#                 module.mask_real.data.fill_(weight_scale)
#
#     # Create the manager object.
#     manager = Manager(args, model)
#
#     # Perform necessary mode operations.
#     if args.mode == 'finetune':
#         if args.no_mask:
#             # No masking will be done, used to run baselines of
#             # Classifier-Only and Individual Networks.
#             # Checks.
#             assert args.lr and args.lr_decay_every
#             assert not args.lr_mask and not args.lr_mask_decay_every
#             assert not args.lr_classifier and not args.lr_classifier_decay_every
#             print('No masking, running baselines.')
#
#             # Get optimizer with correct params.
#             if args.finetune_layers == 'all':
#                 params_to_optimize = model.parameters()
#             elif args.finetune_layers == 'classifier':
#                 for param in model.shared.parameters():
#                     param.requires_grad = False
#                 params_to_optimize = model.classifier.parameters()
#
#             # optimizer = optim.Adam(params_to_optimize, lr=args.lr)
#             optimizer = optim.SGD(params_to_optimize, lr=args.lr,
#                                   momentum=0.9, weight_decay=args.weight_decay)
#             optimizers = Optimizers(args)
#             optimizers.add(optimizer, args.lr, args.lr_decay_every)
#             manager.train(args.finetune_epochs, optimizers,
#                           save=True, savename=args.save_prefix)
#         else:
#             # Masking will be done.
#             # Checks.
#             assert not args.lr and not args.lr_decay_every
#             assert args.lr_mask and args.lr_mask_decay_every
#             assert args.lr_classifier and args.lr_classifier_decay_every
#             print('Performing masking.')
#
#             optimizer_masks = optim.Adam(
#                 model.shared.parameters(), lr=args.lr_mask)#feature extractor layers,不同的classifier任务对应不同的mask
#             optimizer_classifier = optim.Adam(
#                 model.classifier.parameters(), lr=args.lr_classifier)
#
#             optimizers = Optimizers(args)
#             optimizers.add(optimizer_masks, args.lr_mask,
#                            args.lr_mask_decay_every)
#             optimizers.add(optimizer_classifier, args.lr_classifier,
#                            args.lr_classifier_decay_every)
#             manager.train(args.finetune_epochs, optimizers,
#                           save=True, savename=args.save_prefix)
#     elif args.mode == 'eval':
#         # Just run the model on the eval set.
#         manager.eval()
#     elif args.mode == 'check':
#         manager.check()
#
#
# if __name__ == '__main__':
#     main()
