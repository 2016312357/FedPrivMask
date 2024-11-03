#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import math
import numpy as np
import torch
from torch import nn


# from models.dp import alg1


def FedAvg(w, poison_label=list(), w_global=None, return_updates=False):
    '''
    function: 求所有参数更新的平均值
    :param w:
    :return:
    '''

    if len(poison_label) > 0:
        print('aggregation with filtering')
        ww = []
        for i in range(len(poison_label)):
            if poison_label[i] == 'benign':
                ww.append(w[i])

        print(poison_label, len(ww))
        w = ww
    w_avg = copy.deepcopy(w[0])
    if return_updates:
        print('aggregate the model updates, not parameters')
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
        if return_updates:
            w_avg[k] += w_global[k]
    return w_avg

#     perturbed_grads = []
#     for layer in perturbed_dy_dx:
#         layer = layer.to(device)
#         perturbed_grads.append(layer)
#
#     perturbed_gradients_pool.append(perturbed_grads)
#
#
# layers_num = len(gradients_pool[0])
# layer_dims_pool = []
# for layer_gradient in gradients_pool[0]:
#     layer_dims = list((_ for _ in layer_gradient.shape))
#     layer_dims_pool.append(layer_dims)  # shape of each layer
#
# # print(layers_num)
# # print(layer_dims_pool)
#
#
# _gradients = []
# _gradients_perturbed = []
# for layer_index in range(layers_num):
#     gradients__ = torch.zeros(layer_dims_pool[layer_index]).to(device)
#     for gradients_index in range(len(gradients_pool)):
#         gradients__ += gradients_pool[gradients_index][layer_index] \
#                        * aggregation_weight[gradients_index]
#     _gradients.append(gradients__)
#
#     perturbed_gradients__ = torch.zeros(layer_dims_pool[layer_index]).to(device)
#     for gradients_index in range(len(perturbed_gradients_pool)):
#         perturbed_gradients__ += perturbed_gradients_pool[gradients_index][layer_index] \
#                                  * aggregation_weight[gradients_index]
#     _gradients_perturbed.append(perturbed_gradients__)
#
# _scale = 0
# for grad_id in range(aggregation_base):
#     _scale += aggregation_base * perturb_slices_num / slices_num \
#               * (scale ** 2) * aggregation_weight[grad_id]
#
# # Compensate gradients
# gradients_compensated = denoise(gradients=_gradients_perturbed,
#                                 scale=math.sqrt(_scale),
#                                 Q=Q)

# if args.mode.lower() == 'fedbn':  ## no aggregating bn layer
#     for key in server_model.state_dict().keys():
#         if 'bn' not in key:
#             temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
#             for client_idx in range(client_num):
#                 temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
#             server_model.state_dict()[key].data.copy_(temp)
#             for client_idx in range(client_num):
#                 models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
# else:  ##根据loss求客户端更新权重
#     if args.choke and len(train_losses) != 0:
#         loss_mean = np.mean(train_losses)
#         loss_std = np.std(train_losses, ddof=1)
#         if loss_std > 0.2:
#             tmp_total = 0
#             for client_idx in range(len(client_weights)):
#                 if (train_losses[client_idx] > (loss_mean + loss_std)):
#                     client_weights[client_idx] = 0
#                 else:
#                     tmp_total += client_weights[client_idx]
#             client_weights = [client_weights[client_idx] / tmp_total for client_idx in
#                               range(len(client_weights))]
#     for key in server_model.state_dict().keys():
#         # num_batches_tracked is a non-trainable LongTensor and
#         # num_batches_tracked are the same for all clients for the given datasets
#         if 'num_batches_tracked' in key:
#             server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
#         else:
#             temp = torch.zeros_like(server_model.state_dict()[key])
#             for client_idx in range(len(client_weights)):
#                 temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
#             server_model.state_dict()[key].data.copy_(temp)
#             for client_idx in range(len(client_weights)):
#                 models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

from modnets.layers import Sigmoid


def FedAvgMask(model, masks, w, prop):
    """
    function: 求所有参数更新的平均值
    :param masks: mask list for all clients
    :param model: last global model
    :param w: classifier weights, list
    :return:
    """
    mask = copy.deepcopy(masks[0])
    similarity = {}
    for module_idx, module in enumerate(model.modules()):
        # print(module_idx,str(type(module)))
        if 'ElementWise' in str(type(module)):
            baseline = Sigmoid.apply(module.mask_real.data.clone()).reshape(1, -1)
            similarity[module_idx] = []
            similarity[module_idx].append(
                torch.cosine_similarity(mask[module_idx].reshape(1, -1), baseline, dim=-1).item())
            for k in range(1, len(masks)):
                similarity[module_idx].append(
                    torch.cosine_similarity(masks[k][module_idx].reshape(1, -1), baseline, dim=-1).item())
                mask[module_idx] += masks[k][module_idx]
            mask[module_idx] = torch.div(mask[module_idx], len(masks))
            outputs = mask[module_idx].clone()
            outputs.fill_(-0.0005)
            outputs[mask[module_idx] > 0.5] = 0.0005  # mask_real大于0，即sigmoid后大于0.5，baniry mask为1
            module.mask_real.data = outputs
            # print(outputs,'fed')
    # w_avg = copy.deepcopy(w[0])
    # for k in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[k] += w[i][k]
    #     w_avg[k] = torch.div(w_avg[k], len(w))
    # model.classifier.load_state_dict(w_avg)
    return model, similarity


def FedCompare_(model, masks, prop, mask_ori):
    """
    function: 求所有参数更新的平均值
    :param masks: mask list for all clients
    :param model: last global model
    :param w: classifier weights, list
    :return:
    """
    # print(prop)
    flag = False
    mask = copy.deepcopy(masks[0])
    similarity = {'0': {}, '1': {}}  # '0':different property,'1':same
    l1_norm = {'0': {}, '1': {}}  # '0':different property,'1':same
    test_model = {}  # filtered mask
    for module_idx, module in enumerate(model.modules()):
        # print(module_idx,str(type(module)))
        if 'ElementWise' in str(type(module)):
            if isinstance(prop[0], list):
                flag = True
                index = str(int(len(set(prop[0]) & set(prop[1])) > 0))
            else:
                index = str(int(prop[0] == prop[1]))
            # baseline int(prop[0]==prop[1])= Sigmoid.apply(module.mask_real.data.clone()).reshape(1,-1)
            for i in similarity.keys():
                l1_norm[i][module_idx] = []
                similarity[i][module_idx] = []

            l1_norm[index][module_idx].append(((mask[module_idx] - mask_ori[module_idx]).reshape(1, -1) != (
                    masks[1][module_idx] - mask_ori[module_idx]).reshape(1,
                                                                         -1)).sum().item())

            similarity[index][module_idx].append(((mask[module_idx] - mask_ori[module_idx]).reshape(1, -1) != (
                    masks[1][module_idx] - mask_ori[module_idx]).reshape(1,
                                                                         -1)).sum().item() /
                                                 masks[1][module_idx].numel())

            for k in range(len(masks)):
                for j in range(len(masks)):  # for j in range(k+1, len(masks)):
                    # print('property',prop[k],prop[j])
                    if k == j:
                        continue
                    model_key = '{}_{}'.format(k, j)
                    if model_key not in test_model.keys():
                        test_model[model_key] = copy.deepcopy(masks[k])
                    test_model[model_key][module_idx][masks[j][module_idx] == 1] = 0
                    if k == 0 and j == 1:
                        continue
                    if flag:
                        index = str(int(len(set(prop[k]) & set(prop[j])) > 0))
                    else:
                        index = str(int(prop[k] == prop[j]))
                        # print(k,j,index,module_idx,similarity[index][module_idx])
                    similarity[index][module_idx].append(((masks[k][module_idx] - mask_ori[module_idx]).reshape(
                        1, -1) != (masks[j][module_idx] - mask_ori[
                        module_idx]).reshape(1, -1)).sum().item() / masks[1][
                                                             module_idx].numel())
                    l1_norm[index][module_idx].append(((masks[k][module_idx] - mask_ori[module_idx]).reshape(
                        1, -1) != (masks[j][module_idx] - mask_ori[
                        module_idx]).reshape(1, -1)).sum().item())

                    # similarity[index][module_idx].append(torch.cosine_similarity(masks[k][module_idx].reshape(1,-1),
                    #             masks[j][module_idx].reshape(1,-1),dim=-1).item())
                if k != 0:
                    mask[module_idx] += masks[k][module_idx]
            mask[module_idx] = torch.div(mask[module_idx], len(masks))
            outputs = mask[module_idx].clone()
            outputs.fill_(-0.005)
            outputs[mask[module_idx] > 0.5] = 0.005  # mask_real大于0，即sigmoid后大于0.5，baniry mask为1
            module.mask_real.data = outputs

            # avg mask
            mask[module_idx][mask[module_idx] > 0.5] = 1  # mask_real大于0，即sigmoid后大于0.5，baniry mask为1
            mask[module_idx][mask[module_idx] <= 0.5] = 0
            # print((((masks[k][module_idx]-mask_ori[module_idx])==0).sum()/mask[module_idx].numel()).item())
            # print(mask[module_idx].eq(1).sum().item()/mask[module_idx].numel(),'ones:',module_idx)

    return model, similarity, mask, test_model, l1_norm


def FedCompare(model, masks, prop=None, mask_ori=None, epsilon=0, layer_wise=True, poison_label=list(),
               threshold=0.5, return_updates=True):
    """
    function: 求所有参数更新的平均值
    :param masks: mask list for all clients
    :param model: last global model
    :param w: classifier weights, list
    :return:
    """
    if epsilon is not None:
        p = math.exp(epsilon) / (1 + math.exp(epsilon))  # 扰动概率
        print('Aggregation perturbation probability', 1-p)
    else:
        print('no ldp',len(masks))
    ww = []
    if len(poison_label) > 0:
        for i in range(len(poison_label)):
            if poison_label[i] == 'benign':  ##只聚合benign
                ww.append(masks[i])
        masks = ww
        print('aggregation with filtering poison label', poison_label, 'aggregate number', len(masks))
    mask = copy.deepcopy(masks[0])
    similarity = None  # {'0': {}, '1': {}}  # '0':different property,'1':same
    l1_norm = None  # {'0': {}, '1': {}}  # '0':different property,'1':same
    test_model = None  # {}  # filtered mask
    num_perturbation_layer = 0
    for module_idx, module in enumerate(model.modules()):
        if 'ElementWise' in str(type(module)):
            # num_pruned = mask[module_idx].eq(2).sum().item()
            # total = mask[module_idx].numel()
            # # print(module_idx, 'pruned ratio', num_pruned * 1.0 / total)
            weight = torch.ones_like(mask[module_idx])
            num_update = torch.zeros_like(mask[module_idx])
            weight[mask[module_idx] == 2] = 0  # pruned 不参与聚合
            num_update[weight == 1] = 1
            mask[module_idx] = torch.mul(masks[0][module_idx], weight)
            del weight
            for k in range(1, len(masks)):
                weight = torch.ones_like(mask[module_idx])
                weight[masks[k][module_idx] == 2] = 0
                num_update[weight == 1] += 1
                tmp = torch.mul(masks[k][module_idx], weight)
                mask[module_idx] = torch.add(mask[module_idx], tmp)
                # print(mask[module_idx][mask[module_idx]!=0])
                del weight
            if epsilon is None:####<=0

                mask[module_idx][num_update == 0] = 101
                # print('1',module_idx, mask[module_idx][num_update == 0],mask[module_idx][num_update > 0], num_update[num_update != 0])
                mask[module_idx][num_update != 0] = torch.div(mask[module_idx][num_update != 0],
                                                              num_update[num_update != 0])
                if return_updates:
                    print('aggregation and adding updates with thrs', threshold)
                    mask[module_idx][num_update != 0] = torch.add(mask[module_idx][num_update != 0],
                                                              mask_ori[module_idx][num_update != 0])
                # updating the original global model
                # mask[module_idx] = torch.div(mask[module_idx], len(masks))
            elif layer_wise:
                if num_perturbation_layer < 2:
                    num_perturbation_layer += 1
                    mask[module_idx] = (p - 1 + torch.div(mask[module_idx], len(masks))) / (2 * p - 1)
                    # mask[module_idx]=(p-1)/(2*p-1)+mask[module_idx]/((2*p-1)*len(masks))#校正后1的概率
                else:
                    num_perturbation_layer += 1
                    mask[module_idx] = (2 * p - 1) / (2 * 2 * p - 1) + mask[module_idx] / (
                            (2 * 2 * p - 1) * len(masks))  # 校正后1的概率
            elif epsilon>0 and not return_updates:###correct
                mask[module_idx] = (len(masks) * (p - 1) + mask[module_idx]) / (2 * p - 1) / len(masks)
            else:
                print('cannot return updates and apply LDP')
            outputs = mask[module_idx].clone()
            # global binary mask
            mask[module_idx][outputs >= threshold] = 1  # mask_real大于0，即sigmoid后大于0.5，baniry mask为1
            mask[module_idx][outputs < threshold] = 0
            mask[module_idx][outputs == 101] = 2  # mask_real大于0，即sigmoid后大于0.5，baniry mask为1
            print(module_idx, mask[module_idx].eq(2).sum().item() / mask[module_idx].numel(),
                  'rate of none updating avg masks', mask[module_idx].eq(1).sum().item() / mask[module_idx].numel(),
                  '1/all')
            # print((((masks[k][module_idx]-mask_ori[module_idx])==0).sum()/mask[module_idx].numel()).item())
            # print(mask[module_idx].eq(1).sum().item()/mask[module_idx].numel(),'ones:',module_idx)
    return model, similarity, mask, test_model, l1_norm


def compute_dissimilarity(model, masks):  # [task1[{},{},...,{}],task2[],[]]
    similarity = {'0': {}}  # '0':different property,'1':same
    l1_norm = {'0': {}}  # '0':different property,'1':same
    # test_model = {}  # filtered mask
    for module_idx, module in enumerate(model.modules()):
        # print(module_idx,str(type(module)))
        if 'ElementWise' in str(type(module)):
            for i in similarity.keys():
                l1_norm[i][module_idx] = []
                similarity[i][module_idx] = []

            for k in range(len(masks)):  # task
                for u in range(len(masks[0])):  # user id {'layer0':xxx,...}
                    l1_norm['0'][module_idx].append(((masks[k][u][module_idx]) == 1).sum().item())

                for j in range(k + 1, len(masks)):  # for j in range(len(masks)):  #
                    if masks[j][0][module_idx].shape != masks[k][0][module_idx].shape:
                        print('not same model arch', module_idx)
                        continue
                    for u in range(len(masks[0])):
                        similarity['0'][module_idx].append(((masks[k][u][module_idx]).reshape(1, -1) !=
                                                            (masks[j][u][module_idx]).reshape(1, -1)).sum().item() /
                                                           masks[k][u][module_idx].numel())

                        # l1_norm['0'][module_idx].append(((masks[k][u][module_idx]).reshape(
                        #     1, -1) != (masks[j][u][module_idx]).reshape(1, -1)).sum().item())
                        # l1_norm['0'][module_idx].append(((masks[k][u][module_idx])==1).sum().item())

    return similarity, l1_norm
from sklearn.metrics.pairwise import cosine_similarity

def pia_difference(model, masks, mask_ori,return_update=True,args=None,iter=None):  # [{},{},...,{}]
    all_mask = []
    cos_sim=[]
    for k in range(len(masks)):  # task
        mask_diff = {}
        for module_idx, module in enumerate(model.modules()):
            if 'ElementWise' in str(type(module)):
                # masks[k][module_idx][masks[k][module_idx] == 2] = 0
                if not return_update:
                    mask_diff[module_idx] = (masks[k][module_idx] - mask_ori[module_idx]).cpu().numpy() #.reshape(1,-1) # 每一层展开
                else:
                    mask_diff[module_idx][mask_diff[module_idx] == 2] = 0
                    mask_diff[module_idx] = masks[k][module_idx].cpu().numpy()
                if k>0:
                    cos_sim.append(1-cosine_similarity(
                            mask_diff[module_idx].reshape(1, -1), all_mask[0][module_idx].reshape(1, -1))[0][0])
                    with open(f'cosine_{args.data_name}_{args.model}_{args.prop_rate}.txt','a+') as f:
                        f.write(str(iter)+'\t'+str(module_idx)+"\t"+str(cos_sim[-1])+'\n')
                    print(cos_sim[-1])

        all_mask.append(copy.deepcopy(mask_diff))
    print(len(masks), len(all_mask), 'clients',cos_sim)  # [[layer1],[2],..]
    return all_mask


def compute_similarity_poison_fedavg(mask_ori, masks, poison_label):
    similarity = {'malicious': [], 'benign': []}  # '0':different property,'1':same
    for k in range(len(masks)):
        cos_sim = []
        for module_idx in mask_ori.keys():
            if 'running' in module_idx or 'batch' in module_idx or 'bias' in module_idx:
                continue
            # print(module_idx,masks[k][module_idx].shape)
            cos_sim.append(torch.cosine_similarity(
                masks[k][module_idx].reshape(1, -1), mask_ori[module_idx].reshape(1, -1), dim=-1).item())
        cos_sim.append(np.mean(cos_sim))
        similarity[poison_label[k]].append(cos_sim)
    return similarity


def compute_similarity_poison_fedmask(mask_ori, masks, poison_label):
    similarity = {'malicious': [], 'benign': []}  # '0':different property,'1':same
    for k in range(len(masks)):
        cos_sim = []
        for module_idx in mask_ori.keys():
            ### mask only for weights,no bias
            cos_sim.append(torch.cosine_similarity(
                masks[k][module_idx].reshape(1, -1), mask_ori[module_idx].reshape(1, -1), dim=-1).item())
        cos_sim.append(np.mean(cos_sim))
        similarity[poison_label[k]].append(cos_sim)
    return similarity
