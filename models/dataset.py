from __future__ import print_function, division

import tarfile
import urllib

import numpy as np
import torch
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

np.random.seed(1)


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset: 数据集
    :param num_users: 分配给多少用户
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)  # 每个用户平均分到多少数据-600
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        aa = np.random.choice(all_idxs, num_items, replace=False)
        print(i, len(aa))
        dict_users[i] = set(aa)
        # 从总体的all_idxs条数据中选择num_items条数据，不重复
        # numpy.random.choice(a, size=None, replace=True, p=None)
        # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
        # replace:True表示可以取相同数字，False表示不可以取相同数字
        # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
        all_idxs = list(set(all_idxs) - dict_users[i])  # 剩余的数据，保证各个user的数据不重复
    return dict_users  # 返回每个user的数据编号 eg.{user1:(1,3,5,6),user2:(2,4,8,9)...}


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset: 数据集
    :param num_users: 分配给多少用户
    :return: dict of image index
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]  # 0,1,...199
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)  # 60000[0,1,2,...,59999]
    labels = dataset.train_labels.numpy()  # [0,0,...,9,...,9]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 垂直叠加[[],[]]
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 以label聚类，对列进行排序
    idxs = idxs_labels[0, :]  # 只保存数据的索引

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        # 每个user分配2*300条数据索引
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def mnist_property_noiid(dataset, num_users):
    num_shards, num_imgs = 100, 600
    idx_shard = [i for i in range(num_shards)]  # 0,1,...99
    # 索引
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)  # 60000[0,1,2,...,59999]
    labels = dataset.train_labels.numpy()  # [0,0,...,9,...,9]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 垂直叠加[[],[]]
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 以label聚类，对列进行排序
    idxs = idxs_labels[0, :]  # 只保存数据的索引

    # divide and assign
    for i in range(num_users):
        rand_set = idx_shard[i]
        dict_users[i] = np.concatenate((dict_users[i], idxs[rand_set * num_imgs:(rand_set + 1) * num_imgs]), axis=0)

        '''
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        #每个user分配2*300条数据索引
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        '''
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset: 数据集
    :param num_users: 分配给多少用户
    :return: dict of image index
    """
    # num_items = int(len(dataset)/num_users)
    num_items = 10000

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=True))
        all_idxs = list(set(all_idxs))
        # all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, test_dataset, num_users, n_classes_covered, rate_unbalance):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset: 数据集
    :param num_users: 分配给多少用户
    :return: dict of image index
    """
    num_shards, num_imgs = 200, 250  # 1类包含20shards
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000

    idx_shard = [i for i in range(num_shards)]
    label_shard = [i for i in range(10)]
    dict_users = {i: np.array([], dtype='int32') for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}

    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    # stack into two rows, sort the labels row
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]  # only id sorted by labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    label_user = {i: [] for i in range(num_users)}
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_classes_covered, replace=False))  # 用户选中的数据份的id
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                l = labels[rand * num_imgs:(rand + 1) * num_imgs]
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
                user_labels = np.concatenate((user_labels, l),
                                             axis=0)
            else:
                l = labels[rand * num_imgs:int((rand + rate_unbalance) * num_imgs)]
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:int((rand + rate_unbalance) * num_imgs)]),
                    axis=0)
                user_labels = np.concatenate((user_labels, l), axis=0)
            label_user[i].extend(set(l))
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs_test[int(label) * num_imgs_perc_test:int(label + 1) * num_imgs_perc_test]),
                axis=0)

    return dict_users, dict_users_test, label_user


import pandas as pd

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

cm00 = 700  # 234# 202599# 234
cm01 = 700  # 234# 202599# 702
cm10 = 700  # 234# 202599# 702
cm11 = 700  # 234# 202599# 234


# CelebA dataset
class CelebA_property(Dataset):
    def __init__(self, root, label_root, attr='Smiling', transform=None, property='Asian', non=None, iid=True,
                 target_label=-1,loading_num=20000):  # target: attr;  隐私属性: property
        # all attributes
        self.list_attr = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose ' \
                         'Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee ' \
                         'Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard ' \
                         'Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair ' \
                         'Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie ' \
                         'Young'.split()
        # assert attr in self.list_attr
        self.attr = attr
        if isinstance(attr, str):
            target = self.list_attr.index(attr)
        else:
            target = [self.list_attr.index(a) for a in attr]
        print('target label', target)

        if property in self.list_attr:
            privacy = self.list_attr.index(property)
            print('privacy label', privacy)
        else:
            privacy = ''
            print('no privacy label, SR resample')
        # self.transform = transform
        self.root = root

        self.img_dir = os.listdir(root)[:loading_num]  #00 15000[100000:150000]file name
        self.labels = np.load(label_root)  # 按图片id排列
        print(self.labels.shape)
        self.imgs = []
        self.lbs = []
        c00, c11, c10, c01 = 0, 0, 0, 0
        for filename in self.img_dir:
            index = int(filename[: -4]) - 1  # 图片id-1 000001.jpg
            if non is None:  # 不考虑隐私属性分布
                # self.labels[index][target] 
                if isinstance(attr, str):
                    self.lbs.append(int(0) if self.labels[index][target] < 0 else int(1))
                else:
                    self.lbs.append([int(0) if self.labels[index][i] < 0 else int(1) for i in
                                     target])  # int(0) if self.labels[index][target] < 0 else int(1))
                with Image.open(os.path.join(self.root, filename)) as img:
                    self.imgs.append(transform(img))  # tensor
            elif non == 'two':
                assert len(target) == 2
                if c00 >= cm00 and c11 >= cm11 and c10 > cm10 and c01 > cm01:
                    break
                if self.labels[index][target[0]] < 0:
                    if c00 < cm00 and self.labels[index][target[1]] < 0:
                        self.lbs.append([0, 0])
                        with Image.open(os.path.join(self.root, filename)) as img:
                            self.imgs.append(transform(img))
                        c00 += 1
                    elif c01 < cm01 and self.labels[index][target[1]] > 0:
                        self.lbs.append([0, 1])

                        with Image.open(os.path.join(self.root, filename)) as img:
                            self.imgs.append(transform(img))
                        c01 += 1
                elif self.labels[index][target[0]] > 0:
                    if c10 < cm10 and self.labels[index][target[1]] < 0:
                        self.lbs.append([1, 0])
                        with Image.open(os.path.join(self.root, filename)) as img:
                            self.imgs.append(transform(img))
                        c10 += 1
                    elif c11 < cm11 and self.labels[index][target[1]] > 0:
                        self.lbs.append([1, 1])

                        with Image.open(os.path.join(self.root, filename)) as img:
                            self.imgs.append(transform(img))
                        c11 += 1

            elif privacy != '':  ###只选择privacy label=non的sample
                if self.labels[index][privacy] < 0:  # -1,1
                    self.labels[index][privacy] = 0  # 0,1
                if self.labels[index][privacy] == non:  # the attribute label is non
                    if iid == False:  # non-iid
                        print('non iid celeba')
                        if self.labels[index][target] == target_label:  # 1,-1
                            self.lbs.append(int(0) if self.labels[index][target] == -1 else int(1))
                            with Image.open(os.path.join(self.root, filename)) as img:
                                self.imgs.append(transform(img))  # tensor
                    else:  # iid
                        self.lbs.append(int(0) if self.labels[index][target] < 0 else int(1))
                        with Image.open(os.path.join(self.root, filename)) as img:
                            self.imgs.append(transform(img))  # tensor
            else:
                # print('must specify the value of non from {two,0/1,None}')
                exit('must specify the value of non from {two,0/1,None}')
        if non == 'two':
            cmsum = c00 + c11 + c01 + c10
            print("SR:", 1 - 4.0 * ((c00 + c11) * (c01 + c10) / (cmsum * cmsum)), "c00:", c00, "c01:", c01, "c10:", c10,
                  "c11:", c11)
        print('目标属性:', self.attr, 'iid', iid, '隐私属性:', non, property, len(self.imgs), len(self.lbs))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.lbs[idx]


# lfw dataset
class lfw_property(Dataset):
    def __init__(self, root, label_root, attr='Smiling', transform=None, property='Asian', non=1):  # 隐私属性property
        '''
        function: 载入并初始化数据集
        :param root: 数据文件路径
        :param label_root: 标签文件路径
        :param attr: 目标属性
        :param transform: 数据转换方式
        :param property: 敏感属性
        :param non: 敏感属性值
        '''
        self.list_attr = r"person	imagenum	Male	Asian	White	Black	Baby	Child	Young	Middle_Aged	Senior	Black_Hair	Blond_Hair	Brown_Hair	Bald	No_Eyewear	Eyeglasses	Sunglasses	Mustache	Smiling	Frowning	Chubby	Blurry	Harsh Lighting	Flash	Soft Lighting	Outdoor	Curly_Hair	Wavy_Hair	Straight_Hair	Receding_Hairline	Bangs	Sideburns	Fully_Visible_Forehead	Partially_Visible_Forehead	Obstructed_Forehead	Bushy_Eyebrows	Arched_Eyebrows	Narrow_Eyes	Eyes_Open	Big_Nose	Pointy_Nose	Big_Lips	Mouth_Closed	Mouth_Slightly_Open	Mouth_Wide_Open	Teeth_Not_Visible	No_Beard	Goatee	Round_Jaw	Double_Chin	Wearing_Hat	Oval_Face	Square_Face	Round_Face	Color_Photo	Posed_Photo	Attractive_Man	Attractive_Woman	Indian	Gray_Hair	Bags_Under_Eyes	Heavy_Makeup	Rosy_Cheeks	Shiny_Skin	Pale_Skin	5 o'Clock Shadow	Strong Nose-Mouth Lines	Wearing_Lipstick	Flushed_Face	High_Cheekbones	Brown_Eyes	Wearing_Earrings	Wearing_Necktie	Wearing_Necklace".split(
            '\t')  # 数据集文件中所有列名
        # assert attr in self.list_attr  # 判断目标属性是否正确
        self.attr = attr
        attr = pd.read_csv(label_root, delimiter='\t')  # 读取标签文件

        if isinstance(self.attr, str):  # one target attr
            label = np.asarray(attr[self.attr])  # 所有数据对应的目标属性值 within [-1,1]
        else:  # non='two'
            target = [np.asarray(attr[a]) for a in self.attr]  # [[],[]]

        names = np.asarray(attr['person'])  # 名字
        img_num = np.asarray(attr['imagenum'])  # 对应名字的图片id，和名字共同构成图片的文件名

        if property in self.list_attr:
            # privacy = self.list_attr.index(property)
            self.pro = np.asarray(attr[property])  # 所有数据对应的敏感属性值 [-1,1]
        else:
            print('no such privacy attr, please select from',self.list_attr)

        self.imgs = []
        self.labels = []
        index = 0
        if non == 1:  # 提取敏感属性值为正的数据
            for name, num in zip(names, img_num):
                if self.pro[index] <= 0:  # 跳过敏感属性值为负的数据
                    index += 1
                    continue
                name = name.replace(' ', '_')  # name和num共同构成图片的文件名
                with Image.open(os.path.join(root, name, '{}_{}.jpg'.format(name, str(num).zfill(4)))) as img:
                    img = transform(img)  # 数据格式转换为tensor，并作预处理
                    self.imgs.append(img)
                    self.labels.append(int(label[index]>0))
                    index += 1
        elif non == 0:  # 提取敏感属性值为负的数据
            for name, num in zip(names, img_num):
                if self.pro[index] >= 0:  # not of the property
                    index += 1
                    continue
                name = name.replace(' ', '_')
                with Image.open(os.path.join(root, name, '{}_{}.jpg'.format(name, str(num).zfill(4)))) as img:
                    img = transform(img)  # tensor
                    self.imgs.append(img)
                    self.labels.append(int(label[index]>0))###label[index]
                    index += 1
        elif non == 'two':
            c00, c11, c10, c01 = 0, 0, 0, 0
            index = 0
            assert len(self.attr) == 2
            for name, num in zip(names, img_num):
                name = name.replace(' ', '_')  # name和num共同构成图片的文件名
                if c00 >= cm00 and c11 >= cm11 and c10 > cm10 and c01 > cm01:
                    break
                if target[0][index] < 0:  # self.labels[index][target[0]] < 0:
                    if c00 < cm00 and target[1][index] < 0:  # self.labels[index][target[1]] < 0:
                        self.labels.append([0, 0])
                        with Image.open(os.path.join(root, name, '{}_{}.jpg'.format(name, str(num).zfill(4)))) as img:
                            img = transform(img)  # tensor
                            self.imgs.append(img)
                        c00 += 1
                    elif c01 < cm01 and target[1][index] > 0:
                        self.labels.append([0, 1])
                        with Image.open(os.path.join(root, name, '{}_{}.jpg'.format(name, str(num).zfill(4)))) as img:
                            img = transform(img)  # tensor
                            self.imgs.append(img)
                        c01 += 1
                elif target[0][index] > 0:
                    if c10 < cm10 and target[1][index] < 0:
                        self.labels.append([1, 0])
                        with Image.open(os.path.join(root, name, '{}_{}.jpg'.format(name, str(num).zfill(4)))) as img:
                            img = transform(img)  # tensor
                            self.imgs.append(img)
                        c10 += 1
                    elif c11 < cm11 and target[1][index] > 0:
                        self.labels.append([1, 1])
                        with Image.open(os.path.join(root, name, '{}_{}.jpg'.format(name, str(num).zfill(4)))) as img:
                            img = transform(img)  # tensor
                            self.imgs.append(img)
                        c11 += 1
                index += 1
            cmsum = c00 + c11 + c01 + c10
            print("SR:", 1 - 4.0 / (cmsum * cmsum) * (c00 + c11) * (c01 + c10), "c00:", c00, "c01:", c01, "c10:", c10,
                  "c11:", c11)
        print(index, self.attr, '隐私属性:', non, property, len(self.imgs), len(self.labels))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # print(self.imgs[idx].shape,self.labels[idx])#####[3,250,250]
        return self.imgs[idx], self.labels[idx]



class AttackDataset(Dataset):
    '''
    直接将给定的数据封装为攻击数据集
    '''

    def __init__(self, grad_data, label, t=1, train=True, layer='conv1'):
        self.labels = label
        self.flattened_weights = grad_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        flattened_weights = self.flattened_weights[
            index]  # .reshape(-1)  # torch.from_numpy(self.flattened_weights[index]).view(-1)  # .to(device)
        # flattened_weights /= get_1_norm(flattened_weights)
        # print(flattened_weights)
        label = self.labels[index]
        return flattened_weights, label


class Motionsense(Dataset):
    def __init__(self, grad_data, label, priv_labels=None, non=None):
        if non is None:
            self.flattened_weights = grad_data
            self.labels = label

        else:  # non 敏感属性类别
            self.labels = []
            self.flattened_weights = []
            for i in range(len(priv_labels)):
                if priv_labels[i] == non:
                    self.flattened_weights.append(grad_data[i])
                    self.labels.append(label[i])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        flattened_weights = self.flattened_weights[index]
        label = self.labels[index]
        return flattened_weights, label


class Adult(Dataset):
    def __init__(self, data, label, non=None):
        priv_labels=data[:,59:61]
        # print(priv_labels)
        if non is None:
            self.flattened_weights = data
            self.labels = label

        else:  # non 敏感属性类别
            self.labels = []
            self.flattened_weights = []
            for i in range(len(priv_labels)):
                if priv_labels[i][0] == non:
                    self.flattened_weights.append(data[i].astype(np.float32))
                    self.labels.append(int(label[i]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        flattened_weights = self.flattened_weights[index]
        label = self.labels[index]
        return flattened_weights, label

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def show(targets, ret, pro, name='celeba.png'):
    '''
    数据可视化
    :param targets: 属性值列表，不同属性对应不同的颜色
    :param ret: 需要可视化的数据
    :param pro: 可能的属性值名称
    :param name: 保存图片文件名
    :return: None
    '''
    # pro = ['Gray_Hair', 'Smiling', 'Young', 'Male', 'Heavy_Makeup', 'Black_Hair', 'Asian', 'Black', 'White'][:-3]
    colors = ['r', 'g', 'b', 'c', 'y', 'k', 'violet', 'orange', 'purple']  #
    marker = ['<', '*']
    plt.figure(figsize=(5, 5))

    for label in set(targets):  # 0,1
        idx = np.where(np.array(targets) == label)[0]  # [[id,...]]
        plt.scatter(ret[idx, 0], ret[idx, 1], c=colors[label], label=pro[label], marker=marker[label], s=80)
    plt.legend(fontsize=18)
    plt.axes().get_xaxis().set_visible(False)  # 隐藏x坐标轴
    plt.axes().get_yaxis().set_visible(False)  # 隐藏y坐标轴
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.title(name[:-4],fontsize=20)
    plt.savefig('./' + name)


def collect_grads(grads_dict, param_names, avg_pool=True, pool_thresh=5000):
    g = []
    for param_name in param_names:
        grad = grads_dict[param_name]
        # grad = np.asarray(grad)
        shape = grad.shape
        if len(shape) == 1:
            continue
        # grad = np.abs(grad)
        if len(shape) == 4:  # conv
            # if shape[0] * shape[1] > pool_thresh:
            #     continue
            grad = grad.reshape(shape[0], shape[1], -1)
        if len(shape) > 2 or shape[0] * shape[1] > pool_thresh:
            if avg_pool:
                grad = np.mean(grad, -1)
            else:
                grad = np.max(grad, -1)
        g.append(grad.flatten())
    g = np.concatenate(g)#(g, dim=0)  #
    return g


class AttackTotalDataset(Dataset):
    '''
    将给定的数据封装为pytorch识别的数据集格式
    '''

    def __init__(self, grad_data, label, layer, t=1, reduction=False,y=None):
        '''
        :param grad_data: 数据
        :param label: 标签
        :param layer: 需要保留哪些层的数据
        :param t: 需要保留哪些轮次的数据
        :param train: 是否是训练集
        '''
        self.flattened_weights = []
        self.labels = []
        for grad, l in zip(grad_data,
                           label):  # grad is dict {}, 某一个客户端所有层对应的参数更新
            if reduction:
                g = collect_grads(grad, layer, avg_pool=True, pool_thresh=1000)
                # print(g.shape)
                self.flattened_weights.append(g)  # 形如[[],[],[],...] (N,params_numel)
                self.labels.append(l)  # 形如[1,0,1,1,...]
            else:
                # grad: 某一个客户端所有层对应的参数更新（已经展开成(1,param_num)形状），字典类型{'layer_name':[[]]}, l: 0/1数字
                g = grad[layer[0]]  # 暂存第1层参数更新
                # print(len(g),'samples, dataset.py')
                for k in layer[1:]:
                    g = np.concatenate((g, grad[k]), axis=-1)  # 将其他层参数更新拼接到上一层后面
                self.flattened_weights.extend(g)  # 形如[[],[],[],...] (N,params_numel)
                self.labels.extend([l for _ in range(len(g))])  # 形如[1,0,1,1,...]
        # print('attack total dataset shape', len(self.flattened_weights),len(self.labels))
        # tsne = TSNE(n_components=2, init='pca', random_state=42)
        # features = np.vstack(self.flattened_weights)
        # tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
        # import seaborn as sns
        # df = pd.DataFrame()
        # df["y"] = y
        # df["comp1"] = tsne_features[:, 0]
        # df["comp2"] = tsne_features[:, 1]
        # # hue:根据y列上的数据种类，来生成不同的颜色；
        # # style:根据y列上的数据种类，来生成不同的形状点
        # sns.scatterplot(x=df.comp1.tolist(), y=df.comp2.tolist(), hue=df.y.tolist(), style=df.y.tolist(), data=df,
        #                )### markers=True, s=120
        # plt.savefig(os.path.join('./', f"tsne_labels_new.jpg"), format="jpg")
        # plt.show()
        # plt.close()
        # exit(0)
        # if reduction:  # 参数降维；可视化，不同属性对应不同的颜色
        # ret = np.vstack(self.flattened_weights)
        # print('dimention reduction stacking shape', ret.shape)
        # pca = PCA(100, random_state=2304)
        # self.flattened_weights = pca.fit_transform(ret)  # ,t, perplexity=30
        # ret = TSNE(n_components=2, random_state=1).fit_transform(ret)
        #     show(self.labels, ret, ['BlackHair', 'Non-BlackHair'], name='tsne_{}_{}.pdf'.format(layer, t))

        # np.savez('./grad_{}_{}.npz'.format(layer,t),np.vstack(self.flattened_weights),np.asarray(self.labels))

        # self.labels = np.asarray(self.labels)
        # self.flattened_weights = np.vstack(self.flattened_weights)

    def __len__(self):
        assert (len(self.flattened_weights) == len(self.labels))
        return len(self.labels)

    def __getitem__(self, index):
        # print((self.flattened_weights[index]==2).sum().item())

        # self.flattened_weights[index][self.flattened_weights[index]==2]=0
        flattened_weights = self.flattened_weights[index]
        label = self.labels[index]  # torch.Tensor([self.labels[index]])  # .to(device)
        return flattened_weights, label


class Purchase100(Dataset):
    def __init__(self):
        DATASET_PATH = './'  # datasets/purchase
        DATASET_NAME = 'dataset_purchase.npz'
        if not os.path.isdir(DATASET_PATH):
            os.mkdir(DATASET_PATH)
        DATASET_FILE = os.path.join(DATASET_PATH, DATASET_NAME)
        data_set = np.load(DATASET_FILE, allow_pickle=True)['arr_0']
        # print(type(data_set))
        self.X = data_set[:, 1:].astype(np.float32)
        self.Y = ((data_set[:, 0]).astype(np.int32) - 1).astype(np.float32)

        # # if not os.path.isfile(DATASET_FILE):
        # #     # print("Dowloading the dataset...")
        # #     # urllib.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",
        # #     #                    os.path.join(DATASET_PATH, 'tmp.tgz'))
        # #     # print('Dataset Dowloaded')
        # #
        # #     tar = tarfile.open(os.path.join(DATASET_PATH, 'tmp.tgz'))
        # #     tar.extractall(path=DATASET_PATH)
        #
        # data_set = np.loadtxt(DATASET_FILE, delimiter=',')#np.genfromtxt
        #
        # self.X = data_set[:, 1:].astype(np.float64)
        # self.Y = ((data_set[:, 0]).astype(np.int32) - 1).astype(np.float64)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
