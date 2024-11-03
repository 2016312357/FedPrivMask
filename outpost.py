import numpy as np
import torch
import torch.nn as nn


def compute_risk(model: nn.Module):
    var = []
    for param in model.parameters():
        var.append(torch.var(param).cpu().detach().numpy())
    var = [min(v, 1) for v in var]
    return var


def noise(dy_dx: list,phi: float,prune_base: float,noise_base_value: float, risk: list=None):
    if risk is None:
        var = []###layer risk
        for param in dy_dx:
            var.append(torch.var(param).cpu().detach().numpy())
        risk = [min(v, 1) for v in var]
        print('computing',risk)
    # Calculate empirical FIM
    fim = []
    flattened_fim = None
    for i in range(len(dy_dx)):#grad importance
        squared_grad = dy_dx[i].clone().pow(2).mean(0).cpu().numpy()##二范数
        fim.append(squared_grad)
        if flattened_fim is None:
            flattened_fim = squared_grad.flatten()
        else:
            flattened_fim = np.append(flattened_fim, squared_grad.flatten())

    fim_thresh = np.percentile(flattened_fim, 100 - phi)#keep top phi
    for i in range(len(dy_dx)):
        # pruning
        grad_tensor = dy_dx[i].cpu().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, prune_base)#####Config().algorithm.prune_base
        grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
        # noise
        noise_base = torch.normal(0, risk[i] * noise_base_value, dy_dx[i].shape)#Config().algorithm.
        noise_mask = np.where(fim[i] < fim_thresh, 0, 1)
        gauss_noise = noise_base * noise_mask
        dy_dx[i] = (torch.Tensor(grad_tensor) + gauss_noise).to(dtype=torch.float32).to(dy_dx[i].device)

    return dy_dx