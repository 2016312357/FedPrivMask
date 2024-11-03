import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torchvision
from PIL import Image
from torchvision.transforms import transforms
from torchvision.utils import save_image

from sensitivity import compute_sens, compute_sens_all_layer

import inversefed
data_name = 'cifar10'#########'mnist'#

num_images = 1
trained_model = True

if data_name == 'cifar10':
    epochs = 4 #100
    arch = 'ConvNet64'
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative', lr=0.01, epochs=epochs)  ###def.batchsize=64
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders('CIFAR10', defs, '../DATASET/cifar10/')

    model, _ = inversefed.construct_model(arch, num_classes=10, num_channels=3)

    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]

    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
else:
    epochs = 1  ###1#0#100
    arch = 'LeNetZhu'
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative', lr=0.01, epochs=epochs)  ###def.batchsize=64

    loss_fn, trainloader, validloader = inversefed.construct_dataloaders('MNIST', defs, '../DATASET/mnist/')

    model, _ = inversefed.construct_model(arch, num_classes=10, num_channels=1)

    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)
    dm = torch.as_tensor(mnist_mean[0])
    ds = torch.as_tensor(mnist_std[0])

model.to(**setup)



# trans_mnist = transforms.Compose([
#              transforms.ToTensor(), transforms.Normalize(mnist_mean, mnist_std)])
# trainloader = torchvision.datasets.MNIST('../DATASET/mnist/', train=True, download=True,
#                                    transform=trans_mnist)
# testdata = torchvision.datasets.MNIST('../DATASET/mnist/', train=False, download=False,
#                                       transform=trans_mnist)
def plot(tensor, tensor1):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    tensor1 = tensor1.clone().detach()
    tensor1.mul_(ds).add_(dm).clamp_(0, 1)
    print('plotting', tensor.shape)  ###[8, 3, 32, 32]
    from skimage.metrics import structural_similarity as ssim
    # import torchvision.metrics as metrics
    # ssim = metrics.SSIM(data_range=1.0, win_size=11, win_sigma=1.5, k1=0.01, k2=0.03, eps=1e-8, reduction='mean')
    # print(ssim(img1, img2))
    test_ssim = ssim(tensor1.cpu().numpy().squeeze(), tensor.cpu().numpy().squeeze(), channel_axis=0)

    if tensor.shape[0] == 1:
        plt.subplot(2, 1, 1)
        if data_name=='mnist':
            plt.imshow(tensor1[0].permute(1, 2, 0).cpu(),cmap='gray')  #
        else:
            plt.imshow(tensor1[0].permute(1, 2, 0).cpu())  # ,cmap='gray'
        plt.subplot(2, 1, 2)
        if data_name == 'mnist':
            plt.imshow(tensor[0].permute(1, 2, 0).cpu(),cmap='gray')  #
        else:
            plt.imshow(tensor[0].permute(1, 2, 0).cpu())  # ,cmap='gray'
    else:
        fig, axes = plt.subplots(2, tensor.shape[0], figsize=(12, 20))
        for i, im in enumerate(tensor1):
            axes[1][i].imshow(im.permute(1, 2, 0).cpu(), cmap='gray')
        for i, im in enumerate(tensor):
            axes[0][i].imshow(im.permute(1, 2, 0).cpu(), cmap='gray')

    return test_ssim

l={'mnist':[6,25,27],'cifar10':[6,9,10]}#[35,30 ,37,27,25]##1,2,
for idx in l[data_name][:]:#6, 11####6,7  range(25,26)-2
    if trained_model:
        file = f'{arch}_{epochs}.pth'
        try:
            model.load_state_dict(torch.load(f'models/{file}'))
            print('trained model loaded')
        except FileNotFoundError:
            inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup)
            torch.save(model.state_dict(), f'models/{file}')
    model.eval()
    ground_truth, labels = [], []
    # idx = 5 #### 25 # 30###choosen randomly ... just whatever you want
    # img, label = validloader.dataset[idx]  # img, label = testdata[idx]
    # while label != 3:
    #     idx += 1
    #     img, label = validloader.dataset[idx]  # img, label = testdata[idx]
    # labels.append(torch.as_tensor((label,), device=setup['device']))
    # ground_truth.append(img.to(**setup))

    while len(labels) < num_images:
        img, label = validloader.dataset[idx]  # img, label = testdata[idx]
        labels.append(torch.as_tensor((label,), device=setup['device']))
        ground_truth.append(img.to(**setup))
    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)

    local_lr = 1e-4
    local_steps = 1  # local epoch
    use_updates = True

    defense = 'outpost'# 'soteria'#'ours'###'compensate'#'ldp'  #'prune'#''# #''#
    rog=True##attack type
    fc_pruning_rate = 99#50###1e-5###75###1e-6  ##95##0.00005  # #95  ######0.0001##soteria: k%###
    model.zero_grad()
    ground_truth.requires_grad = True

    input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, ground_truth, labels,
                                                                       lr=local_lr, local_steps=local_steps,
                                                                       use_updates=use_updates)
    input_parameters = [p.detach() for p in input_parameters]  ## to be attacked
    # if defense is None:
    if defense == 'prune':
        for i in range(len(input_parameters)):
            grad_tensor = input_parameters[i].cpu().numpy()
            flattened_weights = np.abs(grad_tensor.flatten())
            thresh = np.percentile(flattened_weights, fc_pruning_rate)
            print('pruning layer', i, fc_pruning_rate, 'percentile', thresh)
            grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
            input_parameters[i] = torch.Tensor(grad_tensor).to(**setup)
        # full_norm = torch.stack([g.norm() for g in input_parameters]).mean()
        # print(f'Full gradient norm is {full_norm:e}.')
    elif defense == 'ours':
        for i in range(len(input_parameters)):
            grad_tensor = input_parameters[i].cpu().numpy()
            flattened_weights = np.abs(grad_tensor.flatten())
            thresh = np.percentile(flattened_weights, fc_pruning_rate)
            print('pruning layer', i, fc_pruning_rate, 'percentile', thresh)
            grad_tensor = np.where(abs(grad_tensor) < thresh, 0, 1)
            input_parameters[i] = torch.Tensor(grad_tensor).to(**setup)
        # full_norm = torch.stack([g.norm() for g in input_parameters]).mean()
        # print(f'Full gradient norm is {full_norm:e}.')
    elif defense == 'ldp':
        for i in range(len(input_parameters)):
            grad_tensor = input_parameters[i].cpu().numpy()
            # flattened_weights = grad_tensor.flatten()##np.abs(grad_tensor.flatten())
            noise = np.random.normal(loc=0, scale=fc_pruning_rate,
                                     size=grad_tensor.shape)  # Generate the pruning threshold according to 'prune by percentage'. (Your code: 1 Line)
            grad_tensor += noise
            input_parameters[i] = torch.Tensor(grad_tensor).to(**setup)
    elif defense == 'soteria':
        feature_fc1_graph = model.extract_feature()  ###model(ground_truth)倒数第二层输出展开
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
        print(deviation_f1_x.shape)
        deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
        thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), fc_pruning_rate)
        mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
        input_parameters[-2] = input_parameters[-2] * torch.Tensor(mask).to(**setup)
        print("applying Soteria defense strategy...")
    elif defense == 'compensate':
        print('applying', defense)
        # Q = 6
        slices_num = 10
        scale = 1e-4
        perturb_slices_num = 3
        fc_pruning_rate = [perturb_slices_num / slices_num, scale]
        # Compute layer-wise gradient sensitivity
        sensitivity = compute_sens_all_layer(model=model,
                                             rootset_loader=validloader,  ##64
                                             device=setup['device'])
        from infocom.perturb import noise
        # Slicing gradients and random perturbing
        perturbed_dy_dx = noise(dy_dx=input_parameters,
                                sensitivity=sensitivity,
                                slices_num=slices_num,
                                perturb_slices_num=perturb_slices_num,
                                scale=scale)
        perturbed_grads = []  ### cpu
        for layer in perturbed_dy_dx:
            layer = layer.to(setup['device'])  ### to gpu
            perturbed_grads.append(layer)
        input_parameters = perturbed_grads
    elif defense == 'outpost':
        print('applying', defense)
        phi, prune_base, noise_base=40,80,0.8
        from outpost import noise

        # Slicing gradients and random perturbing
        perturbed_dy_dx = noise(dy_dx=input_parameters,phi=phi,prune_base=prune_base,noise_base_value=noise_base)
        perturbed_grads = []  ### cpu
        for layer in perturbed_dy_dx:
            layer = layer.to(setup['device'])  ### to gpu
            perturbed_grads.append(layer)
        input_parameters = perturbed_grads



    config = dict(signed=True, boxed=True, cost_fn='sim', indices='def', weights='equal',
                  lr=0.0001, optim='adam', restarts=1, max_iterations=28000,  ##1,####10000,#
                  total_variation=1e-6, init='randn', filter='none', lr_decay=True, scoring_choice='loss')

    rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_steps, local_lr, config,
                                                 use_updates=use_updates, num_images=num_images)
    img_shape = (ground_truth.shape[1], ground_truth.shape[2], ground_truth.shape[3])

    output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape=img_shape, ground_truth=ground_truth,rog=rog)

    test_mse = (output.detach() - ground_truth).pow(2).mean()
    feat_mse = (torch.nn.functional.softmax(model(output.detach()), dim=1) - torch.nn.functional.softmax(
        model(ground_truth), dim=1)).pow(2).mean()
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=torch.max(ground_truth))  # 1/ds
    test_ssim = plot(output, ground_truth)

    import lpips
    lpips_metric = lpips.LPIPS(net='alex').to(output.device)
    avg_lpips = lpips_metric(2 * ground_truth - 1, 2 * output.detach() - 1).mean().item()

    print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
          f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4f} | SSIM:{test_ssim:2.4f} | LPIPS: {avg_lpips:2.4f}")

    plt.title(f"MSE:{test_mse:2.4f}|PSNR:{test_psnr:4.2f}|SSIM:{test_ssim:2.4f}|LPIPS:{avg_lpips:2.4f}")

    plt.savefig(f'{data_name}_{defense}_{fc_pruning_rate}_pic_{idx+1}.png')
    print('saving to', f'{data_name}_{defense}_{fc_pruning_rate}_pic_{idx+1}.png')

    # data = inversefed.metrics.activation_errors(model, output, ground_truth)
    plt.show()
    # fig, axes = plt.subplots(2, 3, sharey=False, figsize=(14, 8))
    # axes[0, 0].semilogy(list(data['se'].values())[:-3])
    # axes[0, 0].set_title('SE')
    # axes[0, 1].semilogy(list(data['mse'].values())[:-3])
    # axes[0, 1].set_title('MSE')
    # axes[0, 2].plot(list(data['sim'].values())[:-3])
    # axes[0, 2].set_title('Similarity')
    #
    # convs = [val for key, val in data['mse'].items() if 'conv' in key]
    # axes[1, 0].semilogy(convs)
    # axes[1, 0].set_title('MSE - conv layers')
    # convs = [val for key, val in data['mse'].items() if 'conv1' in key]
    # axes[1, 1].semilogy(convs)
    # convs = [val for key, val in data['mse'].items() if 'conv2' in key]
    # axes[1, 1].semilogy(convs)
    # axes[1, 1].set_title('MSE - conv1 vs conv2 layers')
    # bns = [val for key, val in data['mse'].items() if 'bn' in key]
    # axes[1, 2].plot(bns)
    # axes[1, 2].set_title('MSE - bn layers')
    # fig.suptitle('Error between layers')
    #
    # plt.savefig(f'{data_name}_record.png')
    # plt.show()
