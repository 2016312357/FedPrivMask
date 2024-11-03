from __future__ import division
import matplotlib
import torchvision.models
from sklearn.ensemble import RandomForestClassifier

from generator import generate

matplotlib.use('Agg')
import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, ConcatDataset

from models.attack_models import *
from torch.utils.data import DataLoader

import copy
import numpy as np
from torchvision import transforms

from models.dataset import lfw_property, AttackTotalDataset
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import CNNLfw, ResNet, BasicBlock, VGG11
import torch.nn.functional as F
from models.dataset import AttackDataset

from sklearn.manifold import TSNE
import matplotlib
from sklearn.decomposition import PCA

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_attack(dataloader, net, device, filename=''):
    '''

    :param dataloader: 测试集
    :param net: 被测试模型
    :param device: gpu设备
    :param filename: None
    :return: 准确率
    '''
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total = 0
    correct = 0
    a = []
    l = []

    with torch.no_grad():  # 不计算梯度，加速测试过程
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images).to(device)
            outputs = F.softmax(outputs, dim=-1)
            predicted = torch.argmax(outputs, dim=1)  # (outputs.view(-1) > 0.5).float()
            correct += (predicted == labels).sum().item()
    acc = correct / len(dataloader.dataset)
    # print('PIA test acc {} ({}/{})'.format(acc, correct, len(dataloader.dataset)))
    return acc  # accuracy, precision, recall


def train_attack(dataloader, model, epoch, device, filename='attack_model', dir='./attack_models/'):
    '''
    function: 训练攻击模型
    :param dataloader: 训练集
    :param model: 初始攻击模型
    :param epoch: 训练轮次
    :param device: 设备
    :param filename: 攻击模型保存名称
    :param dir: 保存路径
    :return: None
    '''
    # 定义攻击模型保存位置
    path = os.path.join(dir, f"{filename}.pth")
    if not os.path.exists(dir):
        print('training pia attack model using %d samples'%len(dataloader.dataset))
        os.mkdir(dir)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        print('loading ', path)
        return

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.RMSprop(model.parameters(), lr=0.005, weight_decay=0.00001)
    for ep in range(epoch):
        running_loss = 0
        for i, (inputs, labels) in enumerate(dataloader):
            # print(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)#.reshape()
            optimizer.zero_grad()
            outputs = model(inputs).to(device)
            # print(outputs.shape)
            loss = criterion(outputs, labels.long())  # .squeeze().float()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= (i + 1) # 本轮每个batch的平均损失
        torch.save(model.state_dict(), path)
        if (ep + 1) % 5 == 0:
            print(f'PIA [{ep + 1}] training loss: {running_loss:.3f}', 'saving to', path)

        if running_loss <= 0.001:  # 损失值低于0.001时停止训练
            return

#
# def pia(net_glob, attrs, grad_list, props, args, target, iter):  # attr: 要攻击的属性，props：每条数据对应的属性值列表,iter: global round
#     '''
#     function: layer-wise property inference attack，评估目标模型每一层的属性泄露程度
#     :param net_glob: 目标模型
#     :param attrs: 可能的敏感属性值
#     :param grad_list: 攻击数据
#     :param props: 攻击数据标签
#     :param args: 可用参数
#     :param target: 目标属性
#     :param iter: FL轮次
#     :return: 每一轮的攻击准确率
#     '''
#     for index in range(len(attrs)):
#         attr = attrs[index]
#         # 构造攻击数据标签列表，敏感属性值为index的数据，将其标签设置为1，否则为0
#         prop = [1 if p == index else 0 for p in props]
#         # print(attr, prop)
#         l_all = list(net_glob.state_dict().keys())  # 目标模型所有层名称
#         for layer in l_all:
#             shadow_dataset = AttackDataset(grad_list, prop, layer=layer, t=iter)
#             test_loader = DataLoader(shadow_dataset, batch_size=32, shuffle=False)
#             attack_models = [
#                 AttackNet,
#                 AttackNet1,
#                 AttackNet2,
#                 AttackNet4,
#             ]
#             attack_model = attack_models[0](in_dim=shadow_dataset[0][0].shape[0], out_dim=2).to(args.device)
#             # print(shadow_dataset[0][0].shape[0],len(prop))  # f'{attack_model}'
#             train_attack(None, attack_model, epoch=100, device=args.device,
#                          filename=f'{attack_model}_' + layer + '_task_' + target + '_prop_' + attr,
#                          dir='./reference_model1/')
#             acc = test_attack(test_loader, attack_model, device=args.device)
#             with open('result_ep_{}_task_{}_prop_{}_{}.txt'.format(args.local_ep, target, attrs, args.optim),
#                       'a+') as f:
#                 f.write(attr + '\t' + layer + '\t' + str(iter) + '\t' + str(acc) + '\n')



def pia_shield(layer, grad_list, prop, args, target, privacy, trainset, train_prop, iter, clients_num, m=0):
    '''
    function: 向参数更新中加入对抗扰动
    :param layer: 扰动层名称
    :param grad_list: 待攻击的客户端的参数更新
    :param prop: 待攻击的客户端的敏感属性标签
    :param args: 超参数
    :param target: 目标属性
    :param privacy: 敏感属性
    :param trainset: 攻击模型训练数据
    :param train_prop: 攻击模型训练数据标签
    :param iter: FL global round
    :param clients_num: 客户端总数
    :param m: 本轮参与fl的客户端数目
    :return: 生成的对抗性扰动，扰动后的参数更新
    '''
    if m == 0:
        m = clients_num

    # 准备攻击数据，规范化数据格式
    total_dataset = AttackTotalDataset(trainset, train_prop, layer=layer[:], train=True)  # , t=args.local_ep
    # 划分训练集、验证集
    shadow_dataset, val_dataset = torch.utils.data.random_split(total_dataset, [int(0.6 * len(total_dataset)),
                                                                                len(total_dataset) - int(
                                                                                    0.6 * len(total_dataset))])
    print('pia train on {} samples'.format(len(shadow_dataset)))
    train_loader = DataLoader(shadow_dataset, batch_size=64, shuffle=True)
    # 可选的攻击模型架构
    attack_models = [
        AttackNet,
        AttackNet1,
        AttackNet2,
        AttackNet4,
    ]
    # 定义攻击模型
    attack_model = attack_models[0](in_dim=shadow_dataset[0][0].shape[0], out_dim=len(np.unique(train_prop))).to(
        args.device)  # 'attack class num': len(np.unique(train_prop)))
    if args.data_name == 'CelebA':
        train_attack(train_loader, attack_model, epoch=500, device=args.device,
                     filename=f'{attack_model}_{clients_num}_lr_{args.lr}_round_{args.epochs}_{layer}_layers_task_' + target + '_prop_' + privacy,
                     dir=f'PIA_{args.data_name}'.format())  # (iter-(iter%10))
    elif args.data_name == 'MotionSense':
        train_attack(train_loader, attack_model, epoch=500, device=args.device,
                     filename=f'{attack_model}_{clients_num}_lr_{args.lr}_round_{30}_{layer}_layers_task_' + target + '_prop_' + privacy,
                     dir='PIA_{}'.format(args.data_name))  # (iter-(iter%10))
    else:
        train_attack(train_loader, attack_model, epoch=500, device=args.device,
                     filename=f'{attack_model}_{clients_num}_lr_{args.lr}_round_{30}_{layer}_layers_task_' + target + '_prop_' + privacy,
                     dir='PIA_{}'.format(args.data_name))  # (iter-(iter%10))
    for i in range(iter, iter + 1):
        # 所有客户端本轮的参数更新作为测试集
        g, p = grad_list[(i - 1) * m:i * m], prop[(i - 1) * m:i * m]
        test_dataset = AttackTotalDataset(g, p, layer=layer, train=False)  # , t=args.local_ep
        # 生成对抗性扰动
        perturbed_updates = generate(total_dataset, test_dataset, args.device, attack_model)

        for j in range(m):
            # 保存扰动后的参数更新，便于后续测试
            grad_list[(i - 1) * m + j][layer[0]] += perturbed_updates[j].cpu().numpy()

    return perturbed_updates, grad_list


# attack each single layer
def pia(net_glob, attrs, grad_list, props, args, target, iter):  # attr: 要攻击的属性，props：每条数据对应的属性值列表,iter: global round
    for index in range(len(attrs)):
        attr = attrs[index]
        prop = [1 if p == index else 0 for p in props]
        # print(attr, prop)
        l_all = list(net_glob.state_dict().keys())  # 'conv2.weight'
        for layer in l_all:
            shadow_dataset = AttackDataset(grad_list, prop, layer=layer, t=iter)
            test_loader = DataLoader(shadow_dataset, batch_size=32, shuffle=False)
            attack_models = [
                AttackNet,
                AttackNet1,
                AttackNet2,
                AttackNet4,
            ]
            attack_model = attack_models[0](in_dim=shadow_dataset[0][0].shape[0], out_dim=2).to(args.device)
            # print(shadow_dataset[0][0].shape[0],len(prop))  # f'{attack_model}'
            train_attack(None, attack_model, epoch=100, device=args.device,
                         filename=f'{attack_model}_' + layer + '_task_' + target + '_prop_' + attr,
                         dir='./reference_model/')
            acc = test_attack(test_loader, attack_model, device=args.device)
            with open('result_ep_{}_task_{}_prop_{}_{}.txt'.format(args.local_ep, target, attrs, args.optim),
                      'a+') as f:
                f.write(attr + '\t' + layer + '\t' + str(iter) + '\t' + str(acc) + '\n')




def pia_mask(layer, grad_list, prop, args, target, privacy, trainset, train_prop, iter, clients_num, m=0,labels=None):
    '''
    :function: 实现sp19 Melis's攻击，利用全部层的参数更新进行属性攻击
    :param layer: 利用哪些模型层进行攻击
    :param grad_list: 测试集
    :param prop: 测试集标签
    :param args: 所有可用参数
    :param target: 目标属性
    :param privacy: 敏感属性
    :param trainset: 训练集
    :param train_prop: 训练集标签
    :param iter: FL一共运行的轮次round
    :param clients_num: 客户端总数
    :param m: 本轮参与FL的客户端数目
    :return: 每一轮的攻击准确率（列表形式）
    '''
    # if m == 0:
    #     m = clients_num
    print(len(layer), 'layers are used for PIA, attack prop labels: ',train_prop)
    # clf = RandomForestClassifier(n_estimators=100, n_jobs=5, min_samples_leaf=5, min_samples_split=5)
    #
    # clf.fit(X_train, y_train)
    # y_score = clf.predict_proba(X_test)[:, 1]
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_true=y_test, y_pred=y_pred))
    # print('AUC: ', roc_auc_score(y_true=y_test, y_score=y_score))
    # layer = layer[:3]
    total_dataset = AttackTotalDataset(trainset, train_prop, layer=layer[:],reduction=True,y=labels)  # ,, train=True t=args.local_ep


    # 划分训练集、验证集
    shadow_dataset, val_dataset = torch.utils.data.random_split(total_dataset, [int(0.6 * len(total_dataset)),
                                                                                len(total_dataset) - int(
                                                                                    0.6 * len(total_dataset))])
    # print('pia train on {} samples'.format(len(shadow_dataset)))
    train_loader = DataLoader(shadow_dataset, batch_size=64, shuffle=True)
    # 指定攻击模型架构
    attack_models = [
        AttackNet,
        AttackNet1,
        AttackNet2,
        AttackNet4,
    ]
    attack_model = attack_models[0](in_dim=shadow_dataset[0][0].shape[0], out_dim=len(np.unique(train_prop))).to(
        args.device)
    # print(attack_model)
    # 训练攻击模型
    if args.thrs!=0.5:
        train_attack(train_loader, attack_model, epoch=800, device=args.device,
                     filename=f'{args.data_name}_lr_{args.lr_mask}_round_{args.epochs}_task_{target}_prop_{privacy}_{args.prop_rate}_{args.thrs}',
                     dir='PIA_{}_fedavg_{}'.format(clients_num, args.no_mask, ))  # (iter-(iter%10))

    else:
        train_attack(train_loader, attack_model, epoch=800, device=args.device,
                     filename=f'{args.data_name}_lr_{args.lr_mask}_round_{args.epochs}_task_{target}_prop_{privacy}_{args.prop_rate}',
                     dir='PIA_{}_fedavg_{}'.format(clients_num,args.no_mask))  # (iter-(iter%10))


    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # print('pia test on round of {} on {} clients'.format(i, len(test_dataset)))
    # 第i轮的参数更新的攻击准确率
    # testacc.append(test_attack(test_loader, attack_model, device=args.device))
        
    testacc=test_attack(test_loader, attack_model, device=args.device)
    if grad_list is not None:
        testacc=[testacc]
        for i in range(1, iter + 1):
            # 选取所有客户端第i轮的参数更新作为测试集
            g, p = grad_list[(i - 1) * m:i * m], prop[(i - 1) * m:i * m]
            test_dataset = AttackTotalDataset(g, p, layer=layer[:],reduction=True)  # , t=args.local_ep
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            # 第i轮的参数更新的攻击准确率
            testacc.append(test_attack(test_loader, attack_model, device=args.device))
            print('pia test on round of {} on {} clients'.format(i, len(test_dataset)), testacc[-1])
    return testacc


def pia_all(layer, grad_list, prop, args, target, privacy, trainset, train_prop, iter, clients_num, m=0):
    print(len(layer), 'layers are used for PIA, attack prop labels: ', train_prop)
    total_dataset = AttackTotalDataset(trainset, train_prop, layer=layer[:],
                                       reduction=False)  # ,, train=True t=args.local_ep
    # 划分训练集、验证集
    shadow_dataset, val_dataset = torch.utils.data.random_split(total_dataset, [int(0.6 * len(total_dataset)),
                                                                                len(total_dataset) - int(
                                                                                    0.6 * len(total_dataset))])
    # print('pia train on {} samples'.format(len(shadow_dataset)))
    train_loader = DataLoader(shadow_dataset, batch_size=64, shuffle=True)
    # 指定攻击模型架构
    attack_models = [
        AttackNet,
        AttackNet1,
        AttackNet2,
        AttackNet4,
    ]
    attack_model = attack_models[0](in_dim=shadow_dataset[0][0].shape[0], out_dim=len(np.unique(train_prop))).to(
        args.device)
    # print(attack_model)
    # 训练攻击模型
    train_attack(train_loader, attack_model, epoch=800, device=args.device,
                 filename=f'{args.data_name}_lr_{args.lr_mask}_round_{args.epochs}_task_{target}_prop_{privacy}_{args.prop_rate}',
                 dir='PIA_{}_fedavg_{}'.format(clients_num, args.no_mask,))  # (iter-(iter%10))

    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    testacc = test_attack(test_loader, attack_model, device=args.device)
    if grad_list is not None:
        testacc = [testacc]
        for i in range(1, iter + 1):
            # 选取所有客户端第i轮的参数更新作为测试集
            g, p = grad_list[(i - 1) * m:i * m], prop[(i - 1) * m:i * m]
            test_dataset = AttackTotalDataset(g, p, layer=layer[:], reduction=False)  # , t=args.local_ep
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            # 第i轮的参数更新的攻击准确率
            testacc.append(test_attack(test_loader, attack_model, device=args.device))
            print('pia test on round of {} on {} clients'.format(i, len(test_dataset)),testacc[-1])
        print('pia test average',testacc[0],np.mean(testacc[1:]))
    return testacc


def pia_per_layer(layer, grad_list, prop, args, target, privacy, trainset, train_prop, iter, clients_num, m=0):
    print(len(layer), 'layers are used for PIA, attack prop labels: ', train_prop)
    attack_models = [
        AttackNet,
        AttackNet1,
        AttackNet2,
        AttackNet4,
    ]
    layeracc= {}
    for layerid in range(len(layer)):#####27,
        total_dataset = AttackTotalDataset(trainset, train_prop, layer=layer[layerid:layerid+1],
                                           reduction=False)  # ,, train=True t=args.local_ep
        # 划分训练集、验证集
        shadow_dataset, val_dataset = torch.utils.data.random_split(total_dataset, [int(0.6 * len(total_dataset)),
                                                                                    len(total_dataset) - int(
                                                                                        0.6 * len(total_dataset))])

        train_loader = DataLoader(shadow_dataset, batch_size=64, shuffle=True)
        # 指定攻击模型架构

        attack_model = attack_models[0](in_dim=shadow_dataset[0][0].shape[0], out_dim=len(np.unique(train_prop))).to(
            args.device)
        # 训练攻击模型
        train_attack(train_loader, attack_model, epoch=300, device=args.device,
                     filename=f'{args.model}_id{layerid}_{args.data_name}_lr_{args.lr_mask}_round_{args.epochs}_task_{target}_prop_{privacy}_{args.prop_rate}',
                     dir='PIA_{}_fedavg_{}'.format(clients_num, args.no_mask,))  # (iter-(iter%10))
        test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        testacc = test_attack(test_loader, attack_model, device=args.device)
        if grad_list is not None:
            testacc = [testacc]
            for i in range(1, iter + 1):
                # 选取所有客户端第i轮的参数更新作为测试集
                g, p = grad_list[(i - 1) * m:i * m], prop[(i - 1) * m:i * m]
                test_dataset = AttackTotalDataset(g, p, layer=layer[layerid:layerid+1], reduction=False)  # , t=args.local_ep
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                # 第i轮的参数更新的攻击准确率
                testacc.append(test_attack(test_loader, attack_model, device=args.device))
                print('pia test on round of {} on {} clients'.format(i, len(test_dataset)),testacc[-1])
            print('pia test average',testacc[0],np.mean(testacc[1:]))
        layeracc[layer[layerid]]=testacc
    return layeracc