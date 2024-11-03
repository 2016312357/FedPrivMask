import torch
import time
import os
import copy
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torchvision.transforms import transforms

from datafiles.loaders import dset2loader
from inversefed.nn.models import ConvNet
# from networks import ModifiedDigitModel
import networks as net
from models.Nets import VGG16
from models.dataset import lfw_property, CelebA_property
# from modnets.layers import Sigmoid
from skew import prepare_data, label_skew_across_labels
from datafiles.utils import setseed
from tr_utils import train, train_fedprox, train_LW, test

# cifar10 task
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='test the pretrained model')
parser.add_argument('--percent', type=float, default=0.1, help='no_use,percentage of dataset to train')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--iters', type=int, default=50, help='iterations for communication')
parser.add_argument('--wk_iters', type=int, default=5, help='optimization iters in local worker between communication')
parser.add_argument('--mode', type=str, default='fedprox', help='fedavg | fedprox | fedbn')
parser.add_argument('--mode_agg', type=str, default='', help='fedbn')
parser.add_argument('--affine', type=str, default=False)
###############

parser.add_argument('--mu', type=float, default=0, help='1e-2, 1e-3 0, The hyper parameter for fedprox')
parser.add_argument('--save_path', type=str, default='./checkpoint', help='path to save the checkpoint')
parser.add_argument('--load_path', type=str, default='./checkpoint', help='path to save the checkpoint')
parser.add_argument('--log_path', type=str, default='./log', help='logs_label_weighted, path to save the checkpoint')
parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
parser.add_argument('--choke', action='store_true', help='choke those bad clients when communicating')
parser.add_argument('--label', action='store_true', help='reweight according to label number in FedBN')
parser.add_argument('--model', type=str, default="DigitModel",
                    help='model used:| DigitModel | resnet20 | resnet32 | resnet44 | resnet56 | resnet110 | resnet1202 |')
parser.add_argument('--dataset', type=str, default='mnist', help=' CelebA||mnist| mnist | kmnist | svhn | cifar10 |')
parser.add_argument('--skew', type=str, default='label_across',
                    help='| none | quantity | feat_filter | feat_noise | label_across | label_within |')
parser.add_argument('--Di_alpha', type=float, default=2, help='2,,,0.5,alpha level for dirichlet distribution')
#####mask########
parser.add_argument('--no_mask', default=False,help='Used for running baselines, does not use any masking')
parser.add_argument('--defense', type=str, default='mask')  #compensate mask#ldp #####mask soteria compensate 'ldp' 'soteria'

parser.add_argument('--noise_std', type=float, default=0.5, help='noise level for gaussion noise')
parser.add_argument('--filter_sz', type=int, default=3, help='filter size for filter')
parser.add_argument('--overlap', type=bool, default=True, help='If label_across, allows label distribution to overlap')
parser.add_argument('--nlabel', type=int, default=10, help='number of label for dirichlet label skew')
parser.add_argument('--num_classes', type=int, default=10, help='number of label for dirichlet label skew')
parser.add_argument('--nclient', type=int, default=100, help='client number')
parser.add_argument('--seed', type=int, default=400, help='random seed')

# ###Learning rate
parser.add_argument('--lr_mask', type=float, default=1e-5,
                    help='Learning rate for mask')
parser.add_argument('--lr_mask_decay_every', type=int, default=15,
                    help='Step decay every this many epochs')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Mask threshold')

# Masking options.
parser.add_argument('--mask_init', default='1s',
                    choices=['1s', 'uniform', 'weight_based_1s'],
                    help='Type of mask init')
parser.add_argument('--mask_scale', type=float, default=0.0001,#0.001
                    help='Mask initialization scaling')
parser.add_argument('--mask_scale_gradients', type=str, default='none',
                    choices=['none', 'average', 'individual'],
                    help='Scale mask gradients by weights')
parser.add_argument('--threshold_fn', default='sigmoid',
                    choices=['binarizer', 'ternarizer', 'sigmoid'],
                    help='Type of thresholding function')
args = parser.parse_args()

print(f"args: {args}")

# assert (args.dataset in ['svhn', 'cifar10', 'mnist', 'kmnist'])
assert (args.skew in ['none', 'quantity', 'feat_filter', 'feat_noise', 'label_across', 'label_within'])
assert (args.mode in ['fedavg', 'fedprox', 'fedbn'])

setseed(args.seed)


################# Key Function ########################
def communication(args, server_model, models, client_weights, train_losses, masks_dict=None):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn' or args.mode_agg.lower() == 'fedbn':  ## no aggregating bn layer
            print('aggregation', args.mode)
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    # print('fedbn weights', key, client_weights)
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32,device=device)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(temp)
        else:  ##根据loss求客户端更新权重
            if args.choke and len(train_losses) != 0:
                loss_mean = np.mean(train_losses)
                loss_std = np.std(train_losses, ddof=1)
                if loss_std > 0.2:
                    tmp_total = 0
                    for client_idx in range(len(client_weights)):
                        if (train_losses[client_idx] > (loss_mean + loss_std)):
                            client_weights[client_idx] = 0
                        else:
                            tmp_total += client_weights[client_idx]
                    client_weights = [client_weights[client_idx] / tmp_total for client_idx in
                                      range(len(client_weights))]
            if masks_dict is not None:
                parts=list(masks_dict.keys())
                print('mask aggregation', client_weights,parts)
                mask = copy.deepcopy(masks_dict[parts[0]])##average mask
                similarity = {}
                # tmp_weight = {}
                # tmp_bias = {}
                for module_idx, module in enumerate(server_model.modules()):
                    if 'ElementWise' in str(type(module)):
                        mask[module_idx] = mask[module_idx] * client_weights[parts[0]]
                        for k in parts[1:]:
                            mask[module_idx] += masks_dict[k][module_idx] * client_weights[k]
                        # for k in range(len(client_weights)):
                        #     if k == 0:
                        #         similarity[module_idx] = ((mask[module_idx]).reshape(-1) != (masks_dict[k + 1][module_idx]).reshape(-1)).sum() / len(mask[module_idx].reshape(-1))
                        #         mask[module_idx] = mask[module_idx]*client_weights[k]
                        #         #torch.cosine_similarity(mask[module_idx].reshape(1, -1), masks_dict[k+1][module_idx].reshape(1, -1), dim=-1).item()
                        #     else:
                        #         mask[module_idx] += client_weights[k] * masks_dict[k][module_idx]
                        outputs = mask[module_idx].clone()
                        outputs.fill_(-1)
                        outputs[mask[module_idx] >= 0.6] = 1  # 0.001  # mask_real大于0，即sigmoid后大于0.5，baniry mask为1
                        # print(module_idx, mask[module_idx].shape, outputs.shape, module.mask_real.data.shape)
                        module.mask_real.data.copy_(outputs)
                        mask[module_idx] = outputs
                    # elif 'BatchNorm' in str(type(module)):
                    #     tmp_weight[module_idx] = torch.zeros_like(module.weight)
                    #     tmp_bias[module_idx] = torch.zeros_like(module.weight)
                # print('mask layer-wise dissimilarity', similarity)
                for k in range(len(client_weights)):###updating each client's model including the ones who do not participate
                    for module_idx, module in enumerate(models[k].modules()):
                        if 'ElementWise' in str(type(module)):
                            tmp = torch.mul(module.mask_real.data.abs(), mask[module_idx])
                            module.mask_real.data.copy_(tmp)
            else:
                print('fedavg aggregation',client_weights)
                for key in server_model.state_dict().keys():
                    # num_batches_tracked is a non-trainable LongTensor and
                    # num_batches_tracked are the same for all clients for the given datasets
                    if 'num_batches_tracked' in key:##a same integer
                        server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                    else:
                        temp = torch.zeros_like(server_model.state_dict()[key],device=device,dtype=torch.float32)
                        for client_idx in range(len(client_weights)):
                            temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)##########
                        for client_idx in range(len(client_weights)):
                            models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # print('Device:', device)
    if args.defense == 'mask':
        print('using mask-based training')
        args.no_mask = False


    args.save_path = os.path.join(args.save_path, args.model)
    log_path = os.path.join(args.log_path, args.model)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # if args.choke:
    #     logfile = open(os.path.join(log_path, '{}_{}_{}_{}.log'.format(args.mode + "Choking", args.dataset, args.skew,
    #                                                                    args.nclient)), 'w')
    # elif args.label:
    #     logfile = open(
    #         os.path.join(log_path, '{}_{}_{}_{}_{}.log'.format(args.mode + "Labeling", args.dataset, args.skew,
    #                                                            args.nclient, args.defense)), 'w')
    # else:
    #     logfile = open(os.path.join(log_path,
    #      '{}_{}_{}_{}_Num_{}_{}.log'.format(args.mode, args.dataset, args.skew,args.Di_alpha,args.nclient, args.defense)), 'w')
    #
    # logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    # logfile.write('===Setting===\n')
    # logfile.write('{}\n'.format(args))


    SAVE_PATH = os.path.join(args.save_path, '{}_{}_{}_{}_{}.bin'.format(args.mode, args.dataset, args.skew, args.nclient,args.defense,))
    from models.digit import DigitModel

    if args.no_mask:
        if args.dataset=='mnist':
            server_model = DigitModel(num_classes=10, in_channels=3).to(device)
        else:##cifar10
            # server_model = net.ModifiedResNet(args, mask_init=args.mask_init,
            #                                   mask_scale=args.mask_scale,
            #                                   threshold_fn=args.threshold_fn,
            #                                   original=True).to(device)  # args.no_mask
            server_model = ConvNet(width=64, num_channels=3, num_classes=10).to(device)

    else:
        if args.dataset == 'mnist':
            server_model = DigitModel(num_classes=10, in_channels=3).to(device)
            server_model = net.ModifiedDigitModel(args=args, mask_init=args.mask_init,
                                          mask_scale=args.mask_scale,
                                          threshold_fn=args.threshold_fn,
                                          original=args.no_mask, init=server_model).to(device)
        elif args.dataset == 'CelebA' or args.dataset == 'lfw':  # 指定数据集
            args.num_items_train = 300  # 300 args.num_samples  # 20, number of local data size
            transform = transforms.Compose([  # transforms.Resize([224, 224]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            args.nlabel=2
            args.num_classes = 2
            target = 'Smiling'
            privacy = 'Attractive'#'Eyeglasses'#Black_Hairargs.target[1]  # attr1[5]  #attr1[-2]  # #attr2[0]# attr1[1]#attr1[6] # #指定需要保护的敏感属性
            args.privacy = privacy
            if args.dataset == 'lfw':
                args.fc_input = 25088
                args.data1 = lfw_property('../DATASET/lfw/lfw-deepfunneled/', label_root='../DATASET/lfw/lfw_attributes.txt',
                                     attr=target, transform=transform, property=privacy, non=1)
                # 获取不具有该隐私属性的数据,non=0
                args.data2 = lfw_property('../DATASET/lfw/lfw-deepfunneled/', label_root='../DATASET/lfw/lfw_attributes.txt',
                                     attr=target,
                                     transform=transform, property=privacy, non=0)
            elif args.dataset == 'CelebA':
                args.fc_input = 15360
                args.data1 = CelebA_property('../DATASET/CelebA/img_align_celeba/',
                                        label_root='../DATASET/CelebA/labels.npy', attr=target,
                                        transform=transform, property=privacy, non=1, iid=True)

                args.data2 = CelebA_property('../DATASET/CelebA/img_align_celeba/',
                                        label_root='../DATASET/CelebA/labels.npy', attr=target,
                                        transform=transform, property=privacy, non=0, iid=True)  # eyeglasses

            args.lr_mask = 1e-6
            net_original = VGG16(args).to(device)
            import networks as net
            server_model = net.VGG16Modified(args, mask_init=args.mask_init,
                                         mask_scale=args.mask_scale,
                                         threshold_fn=args.threshold_fn,
                                         original=args.no_mask, init=net_original).to(device)

        else:
            net_original = ConvNet(width=64, num_channels=3, num_classes=10)
            server_model = net.ModifiedConvNet(args=args, mask_init=args.mask_init,
                                                  mask_scale=args.mask_scale,
                                                  threshold_fn=args.threshold_fn,
                                                  original=args.no_mask, init=net_original).to(device)
            # args.arch = 'resnet50'
            # net_original = net.ModifiedResNet(args, mask_init=args.mask_init,
            #                                       mask_scale=args.mask_scale,
            #                                       threshold_fn=args.threshold_fn,
            #                                       original=True).to(device)  # args.no_mask
            #
            # server_model = net.ModifiedResNet(args, mask_init=args.mask_init,
            #    mask_scale=args.mask_scale,threshold_fn=args.threshold_fn,
            #    original=args.no_mask, init=net_original).to(device)
    print(server_model)
    loss_fun = nn.CrossEntropyLoss()
    args.noniid=False#True
    train_loaders, test_loaders = prepare_data(args)# prepare the data  #########
    # #####   warm up ######
    # args.a_iter = 0
    # tr_sets, te_set = label_skew_across_labels(args.dataset, 1, args.nlabel, args.nlabel, False,num_item_per_class=500)
    # tr_l = dset2loader(tr_sets[0], args.batch_size)  # to dataloader
    # optimizer = optim.Adam(params=server_model.parameters(), lr=args.lr_mask, weight_decay=1e-4)  #
    #
    # loss_, acc_, _ = train_fedprox(args, server_model, server_model, tr_l,
    #                optimizer, loss_fun, 1, device,perform_defense=True)
    #
    # del optimizer,tr_l,te_set
    # test_loss, test_acc = test(server_model, test_loader, loss_fun, device)
    # print('warm up Train  Acc: {:.4f}, Test  Acc: {:.4f}\n'.format(acc_,test_acc))

    # federated setting
    client_num = args.nclient

    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn' or args.mode_agg.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = 0##int(checkpoint['a_iter']) + 1
        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0
    # start training
    train_losses = []
    for a_iter in range(resume_iter, args.iters):
        masks_all = {}
        print("============ Train epoch {} ============".format(a_iter))
        participants = np.random.choice(range(client_num), int(client_num * args.percent),replace=False)###np.sort()

        client_weights = [1 / len(participants) if i in participants else 0 for i in range(client_num)]
        samples = [0 for i in range(client_num)]## sample num of each client
        total = 0
        labels = torch.tensor([])
        args.a_iter = a_iter
        for client_idx in participants:#range(client_num):
            model, train_loader, test_loader = models[client_idx], train_loaders[client_idx],test_loaders[client_idx]
            if not args.no_mask:
                optimizer = optim.Adam(params=model.parameters(), lr=args.lr_mask, weight_decay=1e-5)#
            else:
                optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-5)

            if args.mode.lower() == 'fedprox':
                loss_, acc_, masks_all[client_idx] = train_fedprox(args, model, server_model, train_loader,
                                                                   optimizer, loss_fun, client_num, device,
                                                                   perform_defense=True)

            else:
                if args.mode.lower() == 'fedavg':
                    samples[client_idx] += len(train_loader)
                    total += len(train_loader)
                _, _, labels_num = train_LW(model, train_loader, optimizer, loss_fun, client_num, device, args)
                labels = torch.cat((labels, labels_num.unsqueeze(0)), 0)#label num of each client
            # test_loss, test_acc = test(model, test_loader, loss_fun, device)
            # print('Before aggregation, Round {} | client {}| Test loss: {:.4f} | Test  Acc: {:.4f} | Testdata num {}'.format(a_iter, client_idx,test_loss, test_acc,len(test_loader.dataset)))

        # aggregation
        # client_weights = [1 / client_num for i in range(client_num)]
        if args.mode.lower() == 'fedavg':
            client_weights = [samples[i] / total for i in range(client_num)]##没有参与本轮的client仍旧是0
        if (args.mode_agg.lower() == 'fedbn' or args.mode_agg.lower() == 'fedbn') and args.label:
            total = 0
            total_label = torch.sum(labels, dim=0)
            client_w = [0 for i in range(client_num)]
            for i in range(args.nlabel):
                for j in range(client_num):
                    client_w[j] += labels[j][i] / total_label[i]
            client_weights = [client_w[i] / args.nlabel for i in range(client_num)]

        if args.no_mask:
            server_model, models = communication(args, server_model, models,client_weights,train_losses,masks_dict=None)
        else:
            server_model, models = communication(args, server_model, models,client_weights,train_losses,masks_dict=masks_all)

        # min_test_loss = 1000
        max_test_acc = []
        # report after aggregation
        if a_iter==args.iters-1:
            step=1
        else:
            step=10
        for client_idx in range(0,client_num,step):  # , optimizer
            model, train_loader,test_loader = models[client_idx], train_loaders[client_idx],test_loaders[client_idx]  # , optimizers[client_idx]
            train_loss, train_acc = test(model, train_loader, loss_fun, device)
            test_loss, test_acc = test(model, test_loader, loss_fun, device)
            # train_losses.append(train_loss)
            # print(' Round {} | client {}| Train Acc: {:.4f} | Test  Acc: {:.4f} | Testdata num {}'.format(a_iter, client_idx, train_acc,test_acc,len(test_loader.dataset)))
            max_test_acc.append(test_acc)
            # min_test_loss += test_loss
            # logfile.write(
            #     ' Round {} | client {}| Train Loss: {:.4f} | Train Acc: {:.4f} | Test  Acc: {:.4f}\n'.format(a_iter,client_idx, train_loss, train_acc, test_acc))
            # if test_acc > max_test_acc:
            #     server_model = models[client_idx]  ###bn层不一样
            #     max_test_acc = test_acc
            #     min_test_loss = test_loss
        print('Round {} | server | Avg Test  Acc: {:.4f}'.format(a_iter, np.mean(max_test_acc), ))
        # logfile.write('Round {} |  server | Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(a_iter,min_test_loss, max_test_acc))
        # logfile.flush()

    # Save checkpoint
    # print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    # if args.mode.lower() == 'fedbn' or args.mode_agg.lower() == 'fedbn':
    #     dic = {'model_{}'.format(num): models[num].state_dict() for num in range(client_num)}
    #     dic.update({'server_model': server_model.state_dict()})
    #     torch.save(dic, SAVE_PATH)
    # else:
    #     torch.save({
    #         'server_model': server_model.state_dict(),
    #     }, SAVE_PATH)
    del models
    del server_model
    # logfile.flush()
    # logfile.close()
