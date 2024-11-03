# Python version: 3.6
"""
S&P 19 of Melis's property inference attack
Evaluation of Both DP-Laplace & DP-Gaussian
"""
import os
import matplotlib

matplotlib.use('Agg')
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.options import args_parser
from models.attack_models import *

torch.random.manual_seed(1)


class AttackDataset(Dataset):
    def __init__(self, layer=['conv1.weight'], ckp='./checkpoints/', device=None, t=1, train=True):
        # self.device = device
        # self.name = name
        # self.no_index = no_index
        self.flattened_weights = []
        self.labels = []
        # f: with attribute; f1: no attribute
        # model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        print('attacking', ckp)
        data = os.listdir(ckp)
        for record in data:  # round3_client9_prop2_iter22_batch2.pth
            if 'global' in record:
                continue
            d = record.split('_')
            round = int(d[0][5:])
            client_id = int(d[1][6:])
            prop = int(d[2][4:])
            local_iter = int(d[3][4:])
            if train == False:
                # print('only test with epoch', t)
                if round < t or round >= t + 1:  # 只保留t轮的
                    continue
            elif round > t:  # 只保留t轮内的用于训练
                continue
            weights = torch.Tensor().to(device)
            w = torch.load(os.path.join(ckp, record), map_location=torch.device(device))['state_dict'].items()
            for k, v in w:
                if k in layer:
                    weights = torch.cat((weights, v.view(-1).to(device)))

            self.labels.append(prop)
            self.flattened_weights.append(weights)  # %d-%d' %(global_epoch, iter))

        # self.labels = np.asarray(self.labels)
        # self.flattened_weights = np.vstack(self.flattened_weights)
        print('total dataset', (self.flattened_weights[0].shape), len(self.labels))  # , self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        flattened_weights = self.flattened_weights[index].view(
            -1)  # torch.from_numpy(self.flattened_weights[index]).view(-1)  # .to(device)
        label = self.labels[index]  # torch.Tensor([self.labels[index]])  # .to(device)

        # label = torch.Tensor([1.0]) if index < self.positive else torch.Tensor([0.0])
        return flattened_weights, label


def get_eval_metrics(predicted, actual):
    '''
    Return the true positive, true negative, false positive and false negative
    counts given the predicted and actual values.
    '''
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for p, a in zip(predicted, actual):
        if p == 1:
            if a == 1:
                tp += 1
            else:
                fp += 1
        elif p == 0:
            if a == 1:
                fn += 1
            elif a == 0:
                tn += 1
    return tp, tn, fp, fn


def test_attack(dataloader, net, device, filename=''):
    '''
        Evaluate the model on the dataloader and return sigmoid prediction and ground truth labels
                                                        # the accuracy, precision, and recall.
    '''
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total = 0
    correct = 0

    a = []
    l = []

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images).to(device)
            outputs = F.softmax(outputs, dim=-1)
            predicted = torch.argmax(outputs, dim=1)  # (outputs.view(-1) > 0.5).float()
            # print(labels,predicted)#[bs,1]#tensor([0, 0, 1,...], device='cuda:0') tensor([0, 0, 1,...], device='cuda:0')

            # labels = labels#.long().squeeze()
            # print(predicted, labels)  # sigmoid output
            # a.append(outputs.cpu().numpy())  # .view(-1)
            # labels_mul = F.one_hot(labels, len(outputs[0]))
            # l.append(labels_mul.cpu().numpy())

            # try:
            #
            #     total += labels.size(0)
            #
            # except:
            #     print(labels, total)
            #     continue

            correct += (predicted == labels).sum().item()

    correct /= len(dataloader.dataset)
    print('test acc', correct)
    return correct  # accuracy, precision, recall


def train_attack(dataloader, model, epoch, device, filename='attack_model'):
    path = os.path.join('./attack_models/', f"{filename}.pth")
    if os.path.exists(path):
        print('loading done', path)
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        return
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    for ep in range(epoch):
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            # print(inputs,labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).to(device)
            # print(outputs.shape)
            loss = criterion(outputs, labels.long().squeeze())  # .float()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        torch.save(model.state_dict(), path)
        print(f'[{ep + 1}] loss: {running_loss / (i + 1):.3f}')

        if running_loss <= 0.0001:
            return


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.gpu = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    Attr_type = 'Race'
    Target_label = 'Smiling'  # Heavy_Makeup name+ '_' + Attr_type + '_' + Target_label
    attack_models = [
        AttackNet,
        AttackNet1,
        AttackNet2,
        AttackNet4,
    ]
    layer = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight', 'fc3.weight']  #

    args.dp_mechanism = ['laplace', 'gauss'][:]  ### 'none',gauss or laplace ###
    # 设置不加噪的轮次,若为0则每一轮都加噪
    args.noise_free = 0

    args.epsilon = 1
    args.delta = 0.00001
    args.clipthr = 5  # C
    flag = False
    for i in args.dp_mechanism:

        checkpoints = './{}_eps_{}_no_noise_{}/'.format(i, args.epsilon,
                                                        args.noise_free)
        with open('result.txt', 'a+') as f:
            f.write(checkpoints + '\n')
        acc = []
        acc_t = []  # 前t轮的平均攻击成功率

        for t in range(1, args.epochs):  # 按epoch攻击
            shadow_dataset = AttackDataset(layer=layer, ckp=checkpoints, device=args.device, t=t, train=False)
            # print(len(shadow_dataset))

            train_dataset, test_dataset = torch.utils.data.random_split(shadow_dataset,
                                                                        [len(shadow_dataset) - int(
                                                                            0.9 * len(shadow_dataset)),
                                                                         int(0.9 * len(shadow_dataset))])
            print('train:', len(train_dataset), 'test:', len(test_dataset))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            attack_model = attack_models[1](in_dim=shadow_dataset[0][0].shape[0], out_dim=3).to(args.device)
            # print(attack_model)  # f'{attack_model}'
            train_attack(train_loader, attack_model, epoch=100, device=args.device,
                         filename=f'{attack_model}' + '_' + Attr_type + '_' + Target_label)
            acc.append(test_attack(test_loader, attack_model, device=args.device))
            acc_t.append(np.mean(acc))  # 前t轮的平均攻击成功率
        with open('result.txt', 'a+') as f:
            f.write(str(acc) + '\t前t轮的平均攻击成功率\t' + str(acc_t) + '\t Total_avg\t' + str(np.mean(acc)) + '\n')
