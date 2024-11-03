import time

import math
from functools import reduce

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader



def compute_sens_all_layer(model: nn.Module,
                 rootset_loader: DataLoader,
                 device: torch.device,
                 loss_fn = nn.CrossEntropyLoss()):

    x, y = next(iter(rootset_loader))
    # print('x shape is', x.shape)
    # print('y shape is', y.shape)
    x = x.to(device).float().requires_grad_()
    y = y.to(device).long()
    # x = x.to(device).requires_grad_()
    # y = y.to(device)


    # Compute prediction and loss
    pred = model(x)

    loss = loss_fn(pred, y)
    # Backward propagation
    dy_dx = torch.autograd.grad(outputs=loss,
                                inputs=model.parameters(),
                                create_graph=True)
    vector_jacobian_products = []
    for layer in dy_dx:
        # if len(layer.shape)<1:###no noise add to bias of last layer###[:-1]
        #     continue
        # print(layer.shape)
        # Input-gradient Jacobian
        d2y_dx2 = torch.autograd.grad(outputs=layer,
                                      inputs=x,
                                      grad_outputs=torch.ones_like(layer),
                                      retain_graph=True)[0]
        vector_jacobian_products.append(d2y_dx2.detach().clone())

    sensitivity = []
    for layer_vjp in vector_jacobian_products:
        f_norm_sum = 0
        for sample_vjp in layer_vjp:## bs samples

            # Sample-wise Frobenius norm
            f_norm_sum += torch.norm(sample_vjp)
        f_norm = f_norm_sum / len(layer_vjp)
        sensitivity.append(f_norm.cpu().numpy())


    return sensitivity



 # Compute layer-wise gradient sensitivity
def compute_sens(model: nn.Module,
                 rootset_loader: DataLoader, 
                 device: torch.device,
                 loss_fn = nn.CrossEntropyLoss()):
    
    x, y = next(iter(rootset_loader)) # one batch
    # print('x shape is', x.shape)
    # print('y shape is', y.shape)

    x = x.to(device).requires_grad_()
    y = y.to(device)

    # Compute prediction and loss
    pred = model(x)
    loss = loss_fn(pred, y)
    # Backward propagation
    dy_dx = torch.autograd.grad(outputs=loss, 
                                inputs=model.parameters(),
                                create_graph=True)
    dydx=[]
    for name, p in model.named_parameters():
        dydx.append(name)
    vector_jacobian_products = []
    for layer, layer_name in zip(dy_dx, dydx):
        if 'weight' in layer_name:
            # Input-gradient Jacobian
            d2y_dx2 = torch.autograd.grad(outputs=layer, 
                                        inputs=x, 
                                        grad_outputs=torch.ones_like(layer),
                                        retain_graph=True)[0]
            vector_jacobian_products.append(d2y_dx2.detach().clone())
    
    sensitivity = []
    for layer_vjp in vector_jacobian_products:
        f_norm_sum = 0
        for sample_vjp in layer_vjp:
            # Sample-wise Frobenius norm
            f_norm_sum += torch.norm(sample_vjp)
        f_norm = f_norm_sum / len(layer_vjp)
        sensitivity.append(f_norm.cpu().numpy())
    
    return sensitivity


def block_sensitivity(net: nn.Module, rootset_loader: DataLoader, device: torch.device, k: float,
                           return_all=False):
    keys = []
    for key in net.state_dict().keys():
        if 'weight' in key:# or 'bias' in key:
            keys.append(key)

    key_to_num = {key: i for i, key in enumerate(keys)}
    print(keys,key_to_num)

    # Compute the greatest common divisor of parameters of each layer
    num_params_list = []
    for key in net.state_dict().keys():
        if 'weight' in key:
            num_params_list.append(net.state_dict()[key].numel())
    block_size = int(reduce(math.gcd, num_params_list))
    print(f'Block size: {block_size}')

    # 计算block在root dataset的一个batch上的敏感度
    x, y = next(iter(rootset_loader))  # one batch
    x = x.to(device).requires_grad_()
    y = y.to(device)

    pred = net(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(pred, y)
    # 手动backward，得到一阶导数
    dy_dx = torch.autograd.grad(outputs=loss, inputs=net.parameters(), create_graph=True)

    vector_jacobian_products = []

    # TODO: keys有问题，加了batchnorm之后会多出很多有参数但没梯度的层，应该是只有weight和bias有梯度？
    for layer, layer_name in zip(dy_dx, keys):
        if 'weight' in layer_name:
            dy_dx_flattened = torch.flatten(layer)
            print(f'----------{layer_name}----------')
            for i in range(0, len(dy_dx_flattened), block_size):
                dy_dx_flattened_block = dy_dx_flattened[i: i + block_size]
                d2y_dx2 = torch.autograd.grad(outputs=dy_dx_flattened_block, inputs=x,
                                              grad_outputs=torch.ones_like(dy_dx_flattened_block), retain_graph=True)[0]

                vector_jacobian_products.append(d2y_dx2.detach().clone())

    sensitivity = []
    for layer_vjp in vector_jacobian_products:##block-wise
        f_norm_sum = 0
        for sample_vjp in layer_vjp:
            # Sample-wise Frobenius norm
            f_norm_sum += torch.norm(sample_vjp)
        f_norm = f_norm_sum / len(layer_vjp)
        sensitivity.append(f_norm.cpu().numpy())
    return sensitivity
    # blocks = []
    # for name, param in net.named_parameters():
    #     if 'weight' in name:
    #         param = torch.flatten(param)
    #         for i in range(0, len(param), block_size):
    #             block = []
    #             idx = int(i / block_size)
    #             # 换成了数字版的索引
    #             # TODO: reconstruct_image和fed换的时候改
    #             # block.append(key_to_num[name])
    #             block.append(name)
    #             block.append(idx)
    #             block.append(sensitivity[idx])
    #             blocks.append(block)
    #
    # sorted_blocks = sorted(blocks, key=lambda x: x[2])
    #
    # glob_blocks_num = int(len(sorted_blocks) * k)
    # global_blocks = sorted_blocks[0:glob_blocks_num]
    # global_blocks = sorted(global_blocks, key=lambda x: x[0])
    #
    # if return_all:
    #     return global_blocks, sorted_blocks
    # return global_blocks, glob_blocks_num, len(sorted_blocks), block_size


