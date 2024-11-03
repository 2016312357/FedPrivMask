import time

import math
import numpy as np
import torch
from torch.autograd import Variable

from models.dp import clipping
from modnets.layers import Sigmoid
from sensitivity import compute_sens_all_layer


def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    num_data = 0.
    # targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            num_data += target.size(0)
            # targets.append(target.detach().cpu().numpy())
            output = model(data)
            # test_loss += loss_fun(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
    torch.cuda.empty_cache()

    return test_loss / len(test_loader), correct / num_data


def train(model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)

        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all / len(train_iter), correct / num_data


def train_LW(model, train_loader, optimizer, loss_fun, client_num, device, args):
    for wi in range(args.wk_iters):
        model.train()
        num_data = 0
        correct = 0
        loss_all = 0
        labels = torch.tensor([0 for i in range(args.nlabel)])
        train_iter = iter(train_loader)
        for step in range(len(train_iter)):
            optimizer.zero_grad()
            x, y = next(train_iter)
            labels = torch.add(labels, torch.sum(y, dim=0))
            num_data += y.size(0)
            x = x.to(device).float()
            y = y.to(device).long()
            output = model(x)
            loss = loss_fun(output, y)
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
            pred = output.data.max(1)[1]
            correct += pred.eq(y.view(-1)).sum().item()

    return loss_all / len(train_iter), correct / num_data, labels


def train_fedprox(args, model, server_model, train_loader, optimizer, loss_fun, client_num, device,
                  perform_defense=False):
    for wi in range(args.wk_iters):
        model.train()
        num_data = 0
        correct = 0
        loss_all = 0.
        # start = time.time()
        for step, (x, y) in enumerate(train_loader):
            model.zero_grad()
            optimizer.zero_grad()
            num_data += y.size(0)
            x = x.to(device)  # .float()
            y = y.to(device).long()
            output = model(x)
            loss = loss_fun(output, y)
            # for key, param in model.named_parameters():
            #     print(key,param.shape)
            #########################we implement FedProx Here###########################
            # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
            if args.a_iter > 0:
                if args.mu > 0 and wi > 0 and step > 0:  ###
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(server_model.parameters(), model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                        # w_diff += torch.pow(torch.norm(Sigmoid().apply(w) - Sigmoid().apply(w_t)), 2)/w.data.numel()
                        # print(torch.max(w[0]),torch.min(w[0]),w_diff)torch.norm((param - global_weight_collector[param_index]))**2)
                    # print(w_diff,)#Sigmoid().apply(w)
                    loss += args.mu / 2. * w_diff

            loss.backward()
            loss_all += loss.item()
            optimizer.step()
            pred = output.data.max(1)[1]
            correct += pred.eq(y.view(-1)).sum().item()
        # time_used = time.time() - start
        # print(args.a_iter, wi, 'epoch', loss_all / len(train_loader), correct / num_data, 'training samples: ', num_data)
              # 'training samples', time_used/len(train_loader), 'training time used')

    if perform_defense:
        input_parameters = []
        w = model.state_dict()
        for g in w.keys():
            # print(g)
            if 'weight' in g or 'bias' in g:
                input_parameters.append(w[g].detach())
        if args.defense == 'compensate':
            # input_parameters=input_parameters[:-1]
            start = time.time()
            slices_num = 10
            scale = 0.0001
            perturb_slices_num = 3
            # fc_pruning_rate = perturb_slices_num / slices_num
            # Compute layer-wise gradient sensitivity
            sensitivity = compute_sens_all_layer(model=server_model,
                                                 rootset_loader=train_loader, device=device)[:-1]
            from infocom.perturb import noise
            # Slicing gradients and random perturbing all layers, including weights and bias

            ###占用主要的计算时间
            perturbed_dy_dx = noise(dy_dx=input_parameters[:-1],
                                    sensitivity=sensitivity,
                                    slices_num=slices_num,
                                    perturb_slices_num=perturb_slices_num,
                                    scale=scale)
            time_used = time.time() - start
            perturbed_dy_dx.append(input_parameters[-1])
            param_gen = iter(perturbed_dy_dx)
            for g in w.keys():
                if 'weight' in g or 'bias' in g:
                    w[g] = next(param_gen).to(device)
            model.load_state_dict(w)  # 加载加噪后的模型
            del sensitivity, perturbed_dy_dx, param_gen,
            tr_loss, tr_acc = test(model, train_loader, loss_fun, device)
            print('after train acc', tr_acc, args.defense)#, time_used, 'time used')
        elif args.defense == 'soteria':
            # Run reconstruction
            start = time.time()
            fc_pruning_rate = 95
            model.zero_grad()
            ground_truth, labels = x, y
            # ground_truth, labels = ground_truth.to(device), labels.to(device)
            ground_truth.requires_grad = True
            _ = loss_fun(model(ground_truth), labels)
            feature_fc1_graph = model.extract_feature()  # .to(self.args.device)
            deviation_f1_target = torch.zeros_like(feature_fc1_graph)
            deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
            for f in range(deviation_f1_x_norm.size(1)):
                deviation_f1_target[:, f] = 1
                feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)  ###逐层反向传播
                deviation_f1_x = ground_truth.grad.data
                deviation_f1_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
                        feature_fc1_graph.data[:, f] + 0.1)
                model.zero_grad()
                ground_truth.grad.data.zero_()
                deviation_f1_target[:, f] = 0
            deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)  # [512],fc input length
            thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), fc_pruning_rate)
            mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
            input_parameters[-2] = input_parameters[-2] * torch.Tensor(mask).to(device)
            time_used = time.time() - start
            param_gen = iter(input_parameters)
            for g in w.keys():
                if 'weight' in g or 'bias' in g:
                    w[g] = next(param_gen).to(device)
            model.load_state_dict(w)  # 加载加噪后的模型
            tr_loss, tr_acc = test(model, train_loader, loss_fun, device)
            print('after train acc', tr_acc, args.defense, thresh, time_used, 'time used')
        elif args.defense == 'ldp':
            args.dp_mechanism = 'gauss'
            start = time.time()
            # args.epsilon = 20
            # args.delta = 1e-5
            # args.clipthr = 0.01
            # # args.sigma=0.1
            # sensitivity = 2. * args.clipthr / len(train_loader.dataset)  # 计算敏感度S=2*C/N
            # sigma = np.sqrt(2 * np.log(1.25 / args.delta)) * sensitivity / args.epsilon
            sigma = 1e-2  # np.sqrt(2 * np.log(1.25 / args.delta)) * sensitivity / args.epsilon
            # print(args.defense,sigma, args.epsilon, 'sensitivity', sensitivity, 2 * np.log2(1.25 / args.delta))
            # w = clipping(args.clipthr, w)  # 裁剪限定参数大小
            for k in w.keys():
                if 'weight' in k or 'bias' in k:
                    if args.dp_mechanism == 'gauss':
                        noise = np.random.normal(0, sigma, w[k].size())
                        # print('adding noise to', k, noise)
                    elif args.dp_mechanism == 'laplace':
                        noise = np.random.laplace(0, sigma, w[k].size())
                    noise = torch.from_numpy(noise).float().to(device)
                    w[k] += noise
                    # del noise
            time_used = time.time() - start
            model.load_state_dict(w)  # 加载加噪后的模型
            tr_loss, tr_acc = test(model, train_loader, loss_fun, device)
            print('after apply ldp train acc', tr_acc, args.defense, time_used, 'time used')
        elif args.defense == 'prune':
            fc_pruning_rate = 99
            start = time.time()
            for i in range(len(input_parameters)):
                grad_tensor = input_parameters[i].cpu().numpy()
                flattened_weights = np.abs(grad_tensor.flatten())
                thresh = np.percentile(flattened_weights, fc_pruning_rate)
                grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
                input_parameters[i] = torch.Tensor(grad_tensor).to(device)
            print(time.time() - start, 'prune time used')
        elif args.defense == 'mask':
            epsilon = 0########1  ##1  # 3
            masks = {}
            ratio = {}
            if epsilon > 0:
                p = math.exp(epsilon) / (1 + math.exp(epsilon))  # 扰动概率
                print('epsilon,perturbing with', 1 - p)
            # start=time.time()
            for module_idx, module in enumerate(model.modules()):  # shared.
                if 'ElementWise' in str(type(module)):
                    # masks_real[module_idx] = module.mask_real.data.clone()
                    mask = module.mask_real.data  # .cpu()
                    # norm[module_idx] -= copy.deepcopy(mask)  # total mask_real updates
                    # norm[module_idx] = torch.abs(norm[module_idx])
                    # result = torch.sigmoid(mask)
                    outputs = mask.clone()
                    outputs.fill_(0)
                    outputs[mask > 0] = 1  # current binary mask, 0 or 1
                    num_one = outputs.eq(1).sum().item()
                    total = outputs.numel()
                    ratio[module_idx] = num_one * 1.0 / total

                    ######Apply LDP########
                    if epsilon > 0:
                        ran = torch.rand_like(outputs)
                        # print(outputs[ran > p])
                        outputs[ran > p] = 1 - outputs[
                            ran > p]  # flipping# 其正面向上的概率为p，反面向上的概率为1-p。若正面向上，则回答真实答案，反面向上，则回答相反的答案。
                        # print(outputs[ran > p], 'opposite??', )

                    masks[module_idx] = outputs
                # elif 'BatchNorm' in str(type(module)):
                #     masks_real[module_idx] = {}
                #     masks_real[module_idx]['weight'] = module.weight.data
                #     masks_real[module_idx]['bias'] = module.bias.data
                #     masks_real[module_idx]['running_mean'] = module.running_mean
                #     masks_real[module_idx]['running_var'] = module.running_var
            # time_used = time.time() - start
            # del norm
            # print('mask 1/all ratio----------------------------------', ratio)
            tr_loss, tr_acc = test(model, train_loader, loss_fun, device)
            print('after apply defense randomization, train ACC:', tr_acc)#####, time_used, 'time used')###
            return loss_all / len(train_loader), correct / num_data, masks
            # if save:
            #     if not os.path.isdir(self.checkpoints):
            #         os.makedirs(self.checkpoints)
            #     ckpt = {
            #         # 'args': self.args,
            #         # 'ones_ratio':ratio,
            #         'mask': masks,
            #         'norm': norm
            #     }
            #     # Save to file.
            #     torch.save(ckpt,
            #                os.path.join(self.checkpoints,
            #                             savename))  # +savename   os.path.join(self.checkpoints,savename))
            #     print('saving to', self.checkpoints + savename)

    return loss_all / len(train_loader), correct / num_data, None
