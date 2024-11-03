from __future__ import division
# import matplotlib
import numpy as np

'''
Implementation of Membership Inference Attack
'''
# matplotlib.use('Agg')
import os
from torch import nn, optim
from models.attack_models import *
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
from models.dataset import AttackDataset
from sklearn.metrics import roc_auc_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

np.random.seed(12321)


def test_attack(dataloader, net, device):
    '''
    测试成员攻击准确率
    :param dataloader: 数据集
    :param net: 攻击模型
    :param device: gpu/cpu
    :return: 攻击准确率
    '''
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total = 0
    correct = 0
    out = []
    lab = []
    net.eval()
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = torch.sigmoid(net(images)).to(device)  # 输出该数据是成员的概率值
            # print(outputs,labels)
            # outputs = F.softmax(outputs, dim=-1)
            # outputs = torch.sigmoid(outputs)
            predicted = (outputs.view(-1) > 0.5).float()  # 将连续概率值离散化为0/1标签, torch.argmax(outputs, dim=1)  #
            correct += (predicted == labels).sum().item()  # 计算预测成功的个数
            out.extend(outputs.cpu())
            lab.extend(labels.cpu())

    acc = correct / len(dataloader.dataset)  # 计算准确率
    auc = roc_auc_score(lab, out)
    print('Testing MIA acc {} ({}/{})'.format(acc, correct, len(dataloader.dataset)))
    print('Testing MIA auc {} '.format(auc))
    return acc, auc  # accuracy, precision, recall


def train_attack(dataloader, model, epoch, device, filename='attack_model', dir='./attack_models/', dataset=None):
    '''
    训练攻击模型
    :param dataloader: dataloader格式的数据集
    :param model: 攻击模型
    :param epoch: 训练轮次
    :param device: gpu/cpu
    :param filename: 攻击模型文件的名称
    :param dir: 攻击模型保存路径
    :param dataset: dataset格式的原始数据集
    :return:
    '''
    path = os.path.join(dir, f"{filename}.pth")
    if not os.path.exists(dir):
        os.mkdir(dir)
    # if os.path.exists(path):
    #     model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    #     print('loading existed MIA attack model from ', path)
    #     return
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()  # 损失函数
    optimizer = optim.RMSprop(model.parameters(), lr=0.0005, weight_decay=1e-5)  # 优化器
    # optimizer = optim.SGD(model.parameters(), lr=0.00001, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    if dataset is not None:  # 如果给定了数据集，则重新划分训练集和测试集
        d_train, d_test = random_split(dataset, [int(len(dataset) * 0.7), len(dataset) - int(len(dataset) * 0.7)])
        dataloader = DataLoader(d_train, batch_size=64, shuffle=True)
        test_loader = DataLoader(d_test, batch_size=64, shuffle=False)
    else:  # 如果没有给定dataset格式的数据集，则测试集置空
        test_loader = None

    for ep in range(epoch):  # 训练攻击模型
        model.train()
        correct = 0
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            # print(inputs,labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).to(device)
            # print(outputs.shape,labels.shape)
            loss = criterion(outputs.float(), labels.reshape((-1, 1)).float())  # .float()
            loss.backward()
            optimizer.step()
            predicted = (torch.sigmoid(outputs).view(-1) > 0.5).float()  # torch.argmax(outputs, dim=1)  #
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        # torch.save(model.state_dict(), path)
        running_loss /= len(dataloader)
        if (ep + 1) % 25 == 0:
            acc = correct / len(dataloader.dataset)
            print('Epoch {}/{}, Training MIA acc {} ({}/{}), loss {}'.format(ep, epoch, acc, correct,
                                                                             len(dataloader.dataset),
                                                                             running_loss))
            if test_loader is not None:
                acctest = test_attack(test_loader, model, device=device)
                print(f'[{ep + 1}] loss: {running_loss:.3f}, testing acc: {acctest}')

        if running_loss <= 0.01:  # 提前结束训练
            print('return')
            return


def mia(shadow_grad_list, shadow_property_list, grad_list, prop, args, iter=400, clients_num=1):
    """
    function: 实施成员攻击
    :param shadow_grad_list: 攻击模型训练数据
    :param shadow_property_list: 攻击模型训练数据标签
    :param grad_list: 测试数据
    :param prop: 测试标签
    :param args: 所有参数
    :param iter: FL轮次
    :param clients_num: 客户端总数
    :return: 攻击准确率
    """
    test_dataset = AttackDataset(grad_list, prop, t=iter)  # 封装pytorch的测试集
    # 生成成员攻击的训练集
    if shadow_grad_list is not None:
        shadow_dataset = AttackDataset(shadow_grad_list, shadow_property_list, t=iter)
        print('init preparing attack shadow data', shadow_dataset[0][0].shape)
    else:
        shadow_dataset, test_dataset = random_split(test_dataset, [int(len(test_dataset) * 0.6),
                                                                   len(test_dataset) - int(len(test_dataset) * 0.6)])
        print('splitting attack train/test data', shadow_dataset[0][0].shape)
    shadow_loader = DataLoader(shadow_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # 定义攻击模型
    attack_models = [
        AttackNet,
        # AttackNet1,
        # AttackNet2,
        # AttackNet4
    ]
    attack_model = attack_models[-1](in_dim=shadow_dataset[0][0].shape[0], out_dim=1).to(args.device)
    # 开始训练
    init_acc, init_auc = test_attack(test_loader, attack_model, device=args.device)
    if iter > 0:
        train_attack(shadow_loader, attack_model, epoch=iter, device=args.device,
                     filename=f'{attack_model}_{iter}',
                     dir='./MIA_{}/'.format(clients_num))  # ,dataset=shadow_dataset
        # 成员攻击测试
        acc, auc = test_attack(test_loader, attack_model, device=args.device)
    else:
        print('test MIA attack only')
        return init_acc, init_auc
    return acc, auc, init_acc, init_auc
    # with open('MIA_ep_{}.txt'.format(args.local_ep),
    #           'a+') as f:
    #     f.write(str(iter) + '\t' + str(acc) + '\n')

# from utils.options import args_parser
#
# if __name__ == '__main__':
#     args = args_parser()
#     args.mia_ep = 500
#     args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
#     # target = [ 'Young', 'Heavy_Makeup']
#     # target = ['Attractive', 'Smiling']
#     # target = ['Heavy_Makeup', 'Male']
#     # target = ['Blond_Hair', 'Male']
#     # target = ['Attractive', 'Blond_Hair']
#     # target = ['Young', 'Smiling']
#     # target = ['Smiling', 'Male']
#     target = args.target
#     print(target)
#     # attack_x, attack_y = [], []
#     # for i in range(2):
#     #     with np.load(args.save_prefix + '3/test_data_' + target[i] + '.npz') as f:
#     #         attack_x.append(f['arr_0'])
#     #         attack_y.append(f['arr_1'])
#     # # print(attack_x,attack_y)
#     # acc = mia(attack_x[0], attack_y[0], attack_x[1], attack_y[1], args, iter=args.mia_ep)
#     with np.load(args.save_prefix + "11/train_" + target[0] + target[1] + "data" + target[1] + ".npz") as f:
#         grad, label, grad_train, label_train = [f['arr_%d' % i] for i in range(len(f.files))]
#     acc = mia(grad, label, grad_train, label_train, args, iter=args.mia_ep)
