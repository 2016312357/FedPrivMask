#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance):
    num_imgs_train=int(num_samples/n_class)
    print(len(train_dataset), 'training data in total',f'each client holding {num_imgs_train} samples for each class')
    num_shards_train = int(len(train_dataset) / num_imgs_train)
    # 总共num_shards_train份数据，每份大小为num_imgs_train,每个用户从中任选n_class份作为训练集
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = int(len(test_dataset)/num_classes), len(test_dataset)
    assert (n_class <= num_classes)
    assert (n_class * num_users <= num_shards_train)
    # idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train * num_imgs_train)
    labels = np.array(train_dataset.targets)[:len(idxs)]
    # print(len(idxs),len(labels))
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)[:len(idxs_test)]
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  ### sorted labels
    idxs = idxs_labels[0, :]  # index of sorted labels
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    # print(idxs_labels_test[1, :])

    # divide and assign
    label_user = {i: [] for i in range(num_users)}
    for i in range(num_users):
        user_labels = np.array([])
        while len(label_user[i])<n_class:
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))  # 用户选中的数据份的id
            rand=list(rand_set)[0]
            l = labels[rand * num_imgs_train:(rand + 1) * num_imgs_train]

            while list(set(l))[0] in label_user[i]:
                print(list(set(l))[0], label_user[i])
                rand_set = set(np.random.choice(idx_shard, 1, replace=False))  # 用户选中的数据份的id
                rand = list(rand_set)[0]
                l = labels[rand * num_imgs_train:(rand + 1) * num_imgs_train]
            idx_shard = list(set(idx_shard) - rand_set)
            unbalance_flag = 0
            for rand in rand_set:
                if unbalance_flag == 0:  ##该类内数目占其他类数目的unbalance_flag
                    l = labels[rand * num_imgs_train:(rand + 1) * num_imgs_train]
                    dict_users_train[i] = np.concatenate(
                        (dict_users_train[i], idxs[rand * num_imgs_train:(rand + 1) * num_imgs_train]), axis=0)
                    user_labels = np.concatenate((user_labels, l),
                                                 axis=0)
                else:
                    l = labels[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]
                    dict_users_train[i] = np.concatenate(
                        (
                        dict_users_train[i], idxs[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]),
                        axis=0)
                    user_labels = np.concatenate((user_labels, l), axis=0)
                label_user[i].extend(set(l))
                unbalance_flag = 1
            user_labels_set = set(user_labels)


        # unbalance_flag = 0
        # for rand in rand_set:
        #     if unbalance_flag == 0:##该类内数目占其他类数目的unbalance_flag
        #         l = labels[rand * num_imgs_train:(rand + 1) * num_imgs_train]
        #         dict_users_train[i] = np.concatenate(
        #             (dict_users_train[i], idxs[rand * num_imgs_train:(rand + 1) * num_imgs_train]), axis=0)
        #         user_labels = np.concatenate((user_labels, l),
        #                                      axis=0)
        #     else:
        #         l = labels[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]
        #         dict_users_train[i] = np.concatenate(
        #             (dict_users_train[i], idxs[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]),
        #             axis=0)
        #         user_labels = np.concatenate((user_labels, l), axis=0)
        #     label_user[i].extend(set(l))
        #     unbalance_flag = 1
        # user_labels_set = set(user_labels)
        for label in user_labels_set:##测试数据，测试数据中所有该类数据（每类1000条）
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs_test[int(label) * num_imgs_perc_test:int(label + 1) * num_imgs_perc_test]),
                axis=0)
        # print(i,dict_users_train[i])
        # print(set(labels_test_raw[dict_users_test[i].astype(int)]))
    return dict_users_train, dict_users_test, label_user


def mnist_noniid(dataset, num_users, num_samples):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 60000 / num_samples, num_samples  # 100,600
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def celeba_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CelebA dataset
    :param dataset:
    :param num_users:
    :return:
    """

    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
