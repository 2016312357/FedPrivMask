import copy
import torch
from collections import OrderedDict
import numpy as np
import math
from torch.autograd import Variable

def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
def reconstruct_fedavg(args, model, parameters,img_shape,labels=None,reconstruct_label=False,dm=0,ds=1):
    model.to(args.device)
    x_trial=torch.zeros((len(labels), *img_shape),device=args.device)#-0.5)*2)#.to(args.device)
    x_trial.requires_grad = True
    labels = labels.to(args.device)
    local_lossfn=torch.nn.CrossEntropyLoss()
    grad=None
    # optimizer = torch.optim.LBFGS([x_trial])
    model.eval()
    for iter in range(3000):
        # model.zero_grad()
        loss=0
        for _ in range(args.local_ep):
            for batch_id in range(math.ceil(x_trial.shape[0] / args.local_bs)):
                log_probs = model(x_trial[batch_id * args.local_bs:
                                    (batch_id + 1) * args.local_bs])
                labels_ = labels[batch_id * args.local_bs:
                                    (batch_id + 1) * args.local_bs]

                loss += local_lossfn(log_probs, labels_.long()) #mean loss per sample
        loss/=(args.local_ep/(batch_id+1)) 
        grad = torch.autograd.grad(loss, model.parameters(),
                                            retain_graph=True, create_graph=True)
        # print(grad)
        rec_loss = reconstruction_loss(grad, parameters,
                                        rec_lossfn='sim',
                                                indices='def',
                                                weights='equal')
        # print(rec_loss.item())
        rec_loss += 1e-4 * total_variation(x_trial)#噪声图像的梯度变化更大
        # rec_loss.backward()
        x_grad = torch.autograd.grad(rec_loss, [x_trial])[0]
        # print(x_grad.data)
        x_trial.data=x_trial.data-x_grad.data#.sign_()
        del x_grad
        del grad
        # x_trial = torch.clamp(x_trial, 0,1)
        x_trial = torch.clamp(x_trial, (0-dm)/ds, (1-dm)/ds)
        if iter%1000==0:
            print(rec_loss.item())
        
            
    return x_trial
        
        
def reconstruct(args, model, parameters,img_shape,labels=None,reconstruct_label=False):
    pass
#     def _gradient_closure(ori_model, optimizer, x_trial, parameters, labels):
#         def closure():
#             optimizer.zero_grad()#optimize x,y
#             ori_model.zero_grad()
#             ori_model.eval()
#             local_lossfn=torch.nn.CrossEntropyLoss()
#             parameters_trial=None

#             for _ in range(args.local_ep):
#                 # print(x_trial[0])
#                 # x_trial.requires_grad =True
                    
#                 for batch_id in range(max(1,x_trial.shape[0] // args.local_bs)):
#                     # print(x_trial.shape,'reconstruction dlg.py')
#                     log_probs = ori_model(x_trial[batch_id * args.local_bs:
#                                         (batch_id + 1) * args.local_bs])
#                     labels_ = labels[batch_id * args.local_bs:
#                                      (batch_id + 1) * args.local_bs]

#                     loss = local_lossfn(log_probs, labels_.long()) 
#                     grad = torch.autograd.grad(loss, ori_model.parameters(),
#                                                retain_graph=True, create_graph=True, only_inputs=True)
#                     # for id,w in enumerate(parameters): 
#                     #     if id>0:
#                     #         break
#                     #     print(w.data,'before')

#                     # for w, grad_part in zip(ori_model.parameters(),grad):
#                     #     # print(grad_part)
#                     #     w.data = w.data - 0.0001*grad_part                      
                    
#                     # for id,w in enumerate(ori_model.parameters()):
#                     #     if id>0:
#                     #         break
#                     #     print(w.data,'after')
                    
#                     # ori_model.named_parameters() = OrderedDict((name, param - args.lr_mask * grad_part)
#                     #                                        for ((name, param), grad_part)
#                     #                                        in zip(OrderedDict(ori_model.named_parameters()).values(), grad))
                    
#                     if parameters_trial is None:
#                         # parameters_trial=list(OrderedDict((name, param - 0.0001 * grad_part)
#                         #                        for ((name, param), grad_part)
#                         #                        in zip(OrderedDict(ori_model.named_parameters()).items(), grad)).values())
#                         parameters_trial=OrderedDict((name, param - 0.01 * grad_part)
#                                                for ((name, param), grad_part)
#                                                in zip(OrderedDict(ori_model.named_parameters()).items(), grad))#.values()
                    
                    
#                     else:
#                         parameters_trial=OrderedDict((name, param - 0.0001 * grad_part)
#                                                for ((name, param), grad_part)
#                                                in zip(parameters_trial.items(), grad))
#                 # print(batch_id)    
                    
#             parameters_trial = list(parameters_trial.values())

#             rec_loss = reconstruction_loss(parameters_trial, parameters,
#                                             rec_lossfn='l2',
#                                             indices='def',
#                                             weights='equal')#.requires_grad_()
#              # grad = torch.autograd.grad(rec_loss, [x_trial])
#             #
            

#             # if self.rec_config['total_variation'] > 0:
#             rec_loss += 1e-4 * total_variation(x_trial)#噪声图像的梯度变化更大
#             rec_loss.backward()
#             # print(rec_loss.item())
#             # optimizer.step()
            
#             # print(x_trial.grad)
#             # if self.rec_config['signed']:
#             #     x_trial.grad.sign_()
#             return rec_loss
        
        
#         current_loss = closure()
#         # print(x_trial.grad,'before')
#         x_trial.data=x_trial.data-x_trial.grad
        
#         # optimizer.step(closure)
#         # print(x_trial,'after')
#         # print("%.4f" % current_loss.item())
#         return closure
    
#     max_iterations=1000
    
#     model.to(args.device)
#     x_trial=torch.zeros((len(labels), *img_shape),device=args.device)#-0.5)*2)#.to(args.device)
#     x_trial.requires_grad = True
#     # x_trial = Variable(x_trial)
#     labels = labels.to(args.device)
#     parameters = [g.to(args.device) for g in parameters]
#     if reconstruct_label:
#         # print(x_trial.shape,model)
#         output_test = model(x_trial)
#         labels = torch.randn(output_test.shape[1]).to(args.device).requires_grad_(True)
#         # label = Variable(label)
#         # optimizer = torch.optim.Adam([x_trial, labels], lr=0.0001)
#         optimizer = torch.optim.LBFGS([x_trial, labels])
#     else:
#         optimizer = torch.optim.LBFGS([x_trial])
#         # optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)

#     for iteration in range(max_iterations):
#         # optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
#         # print(x_trial[0][0],iteration,x_trial.requires_grad)
#         closure = _gradient_closure(model, optimizer, x_trial, parameters, labels)
#         rec_loss = optimizer.step(closure)

#         if (iteration + 1 == max_iterations) or iteration % 100 == 0:
#             print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')   
#     return x_trial.detach(), labels


def reconstruction_loss(parameters_trial, parameters, rec_lossfn='l2',
                        indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(parameters))
    elif indices == 'batch':
        indices = torch.randperm(len(parameters))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in parameters], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in parameters], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in parameters], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(parameters))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(parameters))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(parameters))[-50:]
    elif indices == 'fc':
        indices = torch.arange(len(parameters))[-2:]
    elif indices == 'rl':
        indices = torch.arange(len(parameters))[:-2]
    elif indices == 'first-conv':
        indices = torch.arange(len(parameters))[:2]
    elif indices == 'all-conv':
        indices = torch.cat((torch.arange(len(parameters))[:-2:4],
                             torch.arange(len(parameters))[1:-2:4]), 0)
    elif indices == 'last2-conv':
        indices = torch.cat((torch.arange(len(parameters))[24:-2:4],
                             torch.arange(len(parameters))[25:-2:4]), 0)
    else:
        raise ValueError()

    setup = parameters[0]
    if weights == 'linear':
        weights = torch.arange(len(parameters), 0, -1, dtype=setup.dtype, device=setup.device) / len(parameters)
    elif weights == 'exp':
        weights = torch.arange(len(parameters), 0, -1, dtype=setup.dtype, device=setup.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = setup.new_ones(len(parameters))

    pnorm = [0, 0]
    loss = 0
    total_loss = 0
    if indices == 'topk-2':
        _, indices = torch.topk(torch.stack([p.norm().detach() for p in parameters_trial], dim=0), 4)
    for i in indices:
        # print(parameters_trial[i].shape,parameters[i].shape)
        if rec_lossfn == 'l2':
            # if i==0:
            #     print(parameters_trial[i],parameters[i])
            loss += ((parameters_trial[i] - parameters[i]).pow(2)).sum() * weights[i]
        elif rec_lossfn == 'l1':
            loss += ((parameters_trial[i] - parameters[i]).abs()).sum() * weights[i]
        elif rec_lossfn == 'max':
            loss += ((parameters_trial[i] - parameters[i]).abs()).max() * weights[i]
        elif rec_lossfn == 'sim':
            loss -= (parameters_trial[i] * parameters[i]).sum() * weights[i]
            pnorm[0] += parameters_trial[i].pow(2).sum() * weights[i]
            pnorm[1] += parameters[i].pow(2).sum() * weights[i]
        elif rec_lossfn == 'simlocal':
            loss += 1 - torch.nn.functional.cosine_similarity(parameters_trial[i].flatten(),
                                                              parameters[i].flatten(),
                                                              0, 1e-10) * weights[i]
    if rec_lossfn == 'sim':
        loss = 1 + loss / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
    total_loss += loss
    return total_loss
