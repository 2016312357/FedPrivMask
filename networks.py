"""Contains various network definitions."""
from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from models.Nets import CNNLfw
import modnets
import modnets.layers as nl


class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

class fc1Modified(nn.Module):
    def __init__(self, args, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False, num_class=1000, init=None):
        super(fc1Modified, self).__init__()
        self.args = args
        self.num_class = num_class
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        # print('threshold', self.threshold_fn)
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original, init)

    def make_model(self, mask_init, mask_scale, threshold_fn, original, init):
        """Creates the model."""
        if original:
            pass
        else:
            # Get the one with masks and pretrained model.
            self.shared = nn.Sequential(
                nl.ElementWiseLinear(self.args.fc_input, 300, mask_init=mask_init, mask_scale=mask_scale,
                                     threshold_fn=threshold_fn),
                nn.ReLU(True),
                nl.ElementWiseLinear(
                    300, 100, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(True),
            )
            self.classifier = nn.Sequential(
                # nl.ElementWiseLinear(4096, num_classes),
                nl.ElementWiseLinear(100, self.args.num_classes, mask_init=mask_init, mask_scale=mask_scale,
                                     threshold_fn=threshold_fn)
            )
            v, u = [], []
            for i in self.shared.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in self.classifier.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in init.modules():
                # print(str(type(i)))
                if 'conv' in str(type(i)) or 'linear' in str(type(i)):
                    u.append(i)
            assert len(u) == len(v)
            for module, module_pretrained in zip(v, u):
                # if isinstance(module, (nl.ElementWiseConv2d, nl.ElementWiseLinear)):
                if 'ElementWise' in str(type(module)):
                    # print('initializing',str(type(module)))
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(fc1Modified, self).train(mode)
        # Set the BNs to eval mode so that the running means and averages do not update.
        # for module in self.shared.modules():
        #     if 'BatchNorm' in str(type(module)):
        #         module.eval()

    def forward(self, x):
        # x = torch.flatten(x, 1)
        x = self.shared(x)
        x = self.classifier(x)
        return x



class ModifiedAlexNet(nn.Module):
    def __init__(self, args, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False, num_class=1000, init=None):
        super(ModifiedAlexNet, self).__init__()
        self.args = args
        self.num_class = args.num_classes
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        # print('threshold', self.threshold_fn)
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original, init)

    def make_model(self, mask_init, mask_scale, threshold_fn, original, init):
        """Creates the model."""
        if original:
            pass
        else:
            # Get the one with masks and pretrained model.
            self.shared = nn.Sequential(
                nl.ElementWiseConv2d(
                    3, 48, kernel_size=11, stride=4, padding=2,
                    mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(inplace=True), #inplace 可以载入更大模型
                nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27] kernel_num为原论文一半
                nl.ElementWiseConv2d(48, 128, kernel_size=5, padding=2,mask_init=mask_init, 
                    mask_scale=mask_scale, threshold_fn=threshold_fn),           # output[128, 27, 27]
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
                nl.ElementWiseConv2d(128, 192, kernel_size=3, padding=1,mask_init=mask_init, 
                    mask_scale=mask_scale, threshold_fn=threshold_fn),          # output[192, 13, 13]
                nn.ReLU(inplace=True),
                nl.ElementWiseConv2d(192, 192, kernel_size=3, padding=1,mask_init=mask_init,
                      mask_scale=mask_scale, threshold_fn=threshold_fn),          # output[192, 13, 13]
                nn.ReLU(inplace=True),
                nl.ElementWiseConv2d(192, 128, kernel_size=3, padding=1,mask_init=mask_init,
                      mask_scale=mask_scale, threshold_fn=threshold_fn),          # output[128, 13, 13]
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)                
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                #全链接128 * 6 * 6
                nl.ElementWiseLinear(self.args.fc_input, 2048, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nl.ElementWiseLinear(2048, 2048, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(inplace=True),
                nl.ElementWiseLinear(2048, self.num_class, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn)
                    # # nl.ElementWiseLinear(4096, num_classes),
                    # nl.ElementWiseLinear(100, self.args.num_classes, mask_init=mask_init, mask_scale=mask_scale,
                    #                      threshold_fn=threshold_fn)
            )
            v, u = [], []
            for i in self.shared.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in self.classifier.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in init.modules():
                # print(str(type(i)))
                if 'conv' in str(type(i)) or 'linear' in str(type(i)):
                    u.append(i)
            assert len(u) == len(v)
            for module, module_pretrained in zip(v, u):
                # if isinstance(module, (nl.ElementWiseConv2d, nl.ElementWiseLinear)):
                if 'ElementWise' in str(type(module)):
                    print('initializing',str(type(module)))
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
            # print('Creating model: Mask layers created.',module_pretrained.weight.data)

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedAlexNet, self).train(mode)
        # Set the BNs to eval mode so that the running means and averages do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def forward(self, x):
        # x = torch.flatten(x, 1)
        x = self.shared(x)
        x = torch.flatten(x, start_dim=1) #展平   或者view()
        x = self.classifier(x)
        return x

class ModifiedDigitModel(nn.Module):
    def __init__(self, args, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False, num_class=10, init=None):
        super(ModifiedDigitModel, self).__init__()
        self.args = args
        self.num_class = args.num_classes
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original, init)

    def make_model(self, mask_init, mask_scale, threshold_fn, original, init):
        """Creates the model."""
        if original:
            pass
        else:
            # Get the one with masks and pretrained model.
            self.shared = nn.Sequential(
                nl.ElementWiseConv2d(
                    3, 64, kernel_size=5, stride=1, padding=2,
                    mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True), #inplace 可以载入更大模型
                nn.MaxPool2d(kernel_size=2),                  #, stride=2 output[48, 27, 27] kernel_num为原论文一半
                nl.ElementWiseConv2d(64, 64, kernel_size=5, stride=1,padding=2,mask_init=mask_init,
                    mask_scale=mask_scale, threshold_fn=threshold_fn),           # output[128, 27, 27]
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),                  # , stride=2output[128, 13, 13]
                nl.ElementWiseConv2d(64, 128, kernel_size=5,stride=1,padding=2,mask_init=mask_init,
                    mask_scale=mask_scale, threshold_fn=threshold_fn),          # output[192, 13, 13]
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True)
            )
            self.classifier = nn.Sequential(
                nl.ElementWiseLinear(128*8*8, 2048, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True),
                nl.ElementWiseLinear(2048, 512, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True),
                nl.ElementWiseLinear(512, self.num_class, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn)
            )
            v, u = [], []
            for i in self.shared.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in self.classifier.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in init.modules():
                # print(str(type(i)))
                if 'conv' in str(type(i)) or 'linear' in str(type(i)):
                    u.append(i)
            assert len(u) == len(v)
            for module, module_pretrained in zip(v, u):
                # if isinstance(module, (nl.ElementWiseConv2d, nl.ElementWiseLinear)):
                if 'ElementWise' in str(type(module)):
                    print('initializing',str(type(module)))
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
            # print('Creating model: Mask layers created.',module_pretrained.weight.data)

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedDigitModel, self).train(mode)
        # Set the BNs to eval mode so that the running means and averages do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()
    def forward(self, x):
        x = self.shared(x)
        # x = x.view(x.shape[0], -1)
        x = torch.flatten(x, start_dim=1) #展平   或者view()
        x = self.classifier(x)
        return x

class CNNLfwModified(nn.Module):  # variant of lenet for CelebA/lfw dataset
    def __init__(self, args, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False, init=None):
        super(CNNLfwModified, self).__init__()
        self.args=args
        # self.num_class = self.args.num_classes
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        print(self.threshold_fn)

        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original,init)

    def make_model(self, mask_init, mask_scale, threshold_fn, original,init):
        """Creates the model."""
        if original:
            pass
            # self.shared = CNNLfw(self.args)  # torchvision
            # print('Creating model: No mask layers.')
        else:
            # Get the one with masks and pretrained model.
            self.shared = nn.Sequential(
                nl.ElementWiseConv2d(
                    3, 6, kernel_size=5,
                    mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(True),
                nl.ElementWiseConv2d(
                    6, 16, kernel_size=5,
                    mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(True)
             )
            self.classifier = nn.Sequential(
                nl.ElementWiseLinear(2704, 120, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(True),
                nl.ElementWiseLinear(
                    120, 84, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(True),
                # nl.ElementWiseLinear(4096, num_classes),
                nl.ElementWiseLinear(84, self.args.num_classes, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn)
            )
            v,u=[],[]
            for i in self.shared.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in self.classifier.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in init.modules():
                if 'conv' in str(type(i)) or 'linear' in str(type(i)):
                    u.append(i)
            assert len(u)==len(v)
            for module, module_pretrained in zip(v, u):
                if isinstance(module, (nl.ElementWiseConv2d, nl.ElementWiseLinear)):
                    # if 'ElementWise' in str(type(module)):
                    # print(str(type(module)))
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
            # print('Creating model: Mask layers created.',module_pretrained.weight.data)

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(CNNLfwModified, self).train(mode)
        # Set the BNs to eval mode so that the running means and averages do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()
    def forward(self, x):
        x = self.shared(x)
        # x = x.view(x.size(0),-1)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.classifier(x)
        return x


class CNNMnistModified(nn.Module):  # variant of lenet for CelebA/lfw dataset
    def __init__(self, args, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False,init=None):
        super( CNNMnistModified, self).__init__()
        self.args=args
        # self.num_class = args.num_classes
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        # print(self.threshold_fn)
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original,init)

    def make_model(self, mask_init, mask_scale, threshold_fn, original,init):
        """Creates the model."""
        if original:
            pass
            # self.shared = CNNLfw(self.args)  # torchvision
            # print('Creating model: No mask layers.')
        else:
            # Get the one with masks and pretrained model.
            self.shared = nn.Sequential(
                nl.ElementWiseConv2d(
                    1, 32,kernel_size=3,stride=1,padding=1,
                    mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2),
                
                nl.ElementWiseConv2d(
                    32,64,kernel_size=3,stride=1,padding=1,
                    mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2),
                
             )
            self.classifier = nn.Sequential(
                nl.ElementWiseLinear(64*7*7,1024, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(True),
                nl.ElementWiseLinear(
                    1024,512, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(True),
                # nl.ElementWiseLinear(4096, num_classes),
                nl.ElementWiseLinear(512, self.args.num_classes, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn)
            )
            v,u=[],[]
            for i in self.shared.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in self.classifier.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in init.modules():
                if 'conv' in str(type(i)) or 'linear' in str(type(i)):
                    u.append(i)
            assert len(u)==len(v)
            for module, module_pretrained in zip(v, u):
                # if isinstance(module, (nl.ElementWiseConv2d, nl.ElementWiseLinear)):
                if 'ElementWise' in str(type(module)):
                    print('initializing',str(type(module)))
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
            # print('Creating model: Mask layers created.',module_pretrained.weight.data)

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(CNNMnistModified, self).train(mode)
        # Set the BNs to eval mode so that the running means and averages do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()
    def forward(self, x):
        x = self.shared(x)
        # x = x.view(x.size(0),-1)
        x = x.view(-1, 64 * 7* 7)#将数据平整为一维的 
        
        # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.classifier(x)
        return x


class VGG16Modified(nn.Module):  # for CelebA/lfw dataset
    def __init__(self, args, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False,init=None):
        super(VGG16Modified, self).__init__()
        self.args=args
        # self.num_class = args.num_classes
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original,init)

    def make_model(self, mask_init, mask_scale, threshold_fn, original,init):
        """Creates the model."""
        if original:
            print('Warning Creating model: No mask layers.')
            exit(0)
        else:
            self.shared = nn.Sequential(
                    nl.ElementWiseConv2d(3, 64, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            ###track_running_stats设为True时，BatchNorm层会统计更新全局的均值running_mean和方差running_var 
            ###affine设为True时，BatchNorm层才会学习参数\gamma和\beta，否则不包含变量weight和bias 
            nn.ReLU(inplace=True),
                nl.ElementWiseConv2d(64, 64, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, ),        
            
                nl.ElementWiseConv2d(64, 128, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
                nl.ElementWiseConv2d(128, 128, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, ),
        
                nl.ElementWiseConv2d(128, 256, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
                nl.ElementWiseConv2d(256, 256, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
                nl.ElementWiseConv2d(256, 256, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, ),
            
            
            
            nl.ElementWiseConv2d(256, 512, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
                nl.ElementWiseConv2d(512, 512, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
                nl.ElementWiseConv2d(512, 512, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, ),


                nl.ElementWiseConv2d(512, 512, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
                nl.ElementWiseConv2d(512, 512, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
                nl.ElementWiseConv2d(512, 512, kernel_size=3, stride=1, padding=1,mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )
            self.classifier = nn.Sequential(
                    # nn.Dropout(p=0.5),
                    # nn.Linear(4096, 4096, bias=True),
                    # nn.ReLU(inplace=True),
                    # nn.Dropout(p=0.5),
                    # nn.Linear(4096, 2, bias=True),
                nl.ElementWiseLinear(self.args.fc_input, 4096, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(True),
                nl.ElementWiseLinear(
                    4096, 4096, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(True),
                # nl.ElementWiseLinear(4096, num_classes),
                nl.ElementWiseLinear(4096, self.args.num_classes, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn)
            )
            v,u=[],[]
            for i in self.shared.modules():
                if 'ElementWise' in str(type(i)) or 'BatchNorm' in str(type(i)):
                    v.append(i)
            for i in self.classifier.modules():
                if 'ElementWise' in str(type(i)) or 'BatchNorm' in str(type(i)):
                    v.append(i)
            for i in init.modules():
                if 'conv' in str(type(i)) or 'linear' in str(type(i)) or 'batchnorm' in str(type(i)):
                    u.append(i)
            # print(len(u),len(v))
            # for module, module_pretrained in zip(v, u):
            #     print(str(type(module)),str(type(module_pretrained)))
            assert len(u)==len(v)
            for module, module_pretrained in zip(v, u):
                print(str(type(module)),str(type(module_pretrained)))
                # if isinstance(module, (nl.ElementWiseConv2d, nl.ElementWiseLinear)):
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    # std=module_pretrained.weight.data.abs().cpu().mean()
                    # module.mask_real.data[module_pretrained.weight.data.abs().cpu()>std]=0.01
                    # print(module.mask_real.data,module.mask_real.data.shape,module_pretrained.weight.shape,'initializing',str(type(module)),std)
                elif 'BatchNorm' in str(type(module)):
                    if module_pretrained.weight is not None:
                        module.weight.data.copy_(module_pretrained.weight.data)
                        module.bias.data.copy_(module_pretrained.bias.data)
                    if module.running_mean is not None:
                        module.running_mean.copy_(module_pretrained.running_mean)
                        module.running_var.copy_(module_pretrained.running_var)
            print('Creating model: Mask BN layers created.')
            # print('Creating model: Mask layers created.',module_pretrained.weight.data)

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(VGG16Modified, self).train(mode)
        # Set the BNs to eval mode so that the running means and averages do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()
    def forward(self, x):
        x = self.shared(x)###
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

class ModifiedVGG16(nn.Module):
    """VGG16 with support for multiple classifiers."""
    def __init__(self, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False, num_class=1000):
        super(ModifiedVGG16, self).__init__()
        self.num_class = num_class
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original)

    def make_model(self, mask_init, mask_scale, threshold_fn, original):
        """Creates the model."""
        if original:
            vgg16 = models.vgg16(pretrained=True)  # torchvision
            # print('Creating model: No mask layers.')
        else:
            # Get the one with masks and pretrained model.
            vgg16 = modnets.vgg16(mask_init, mask_scale, threshold_fn)
            vgg16_pretrained = models.vgg16(pretrained=True)  # 1000
            # print(vgg16,vgg16_pretrained)
            # Copy weights from the pretrained to the modified model.
            v=[]
            for i in vgg16.modules():
                if 'Sigmoid' not in str(type(i)) and 'View' not in str(type(i)):
                    v.append(i)
            for module, module_pretrained in zip(v, vgg16_pretrained.modules()):
                # print(str(type(module)), str(type(module_pretrained)))
                # for module, module_pretrained in zip(vgg16.children(), vgg16_pretrained.children()):
                if isinstance(module, (nl.ElementWiseConv2d, nl.ElementWiseLinear)):
                    # if 'ElementWise' in str(type(module)):
                    # print(str(type(module)))
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)


        self.datasets, self.classifiers = [], nn.ModuleList()

        idx = 6
        for module in vgg16.classifier.children():
            if isinstance(module, (nn.Linear, nl.ElementWiseLinear)):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)  # 只包含最后一个全连接输出层，其余全连接层都放在features里
                idx += 1
        features = list(vgg16.features.children())
        features.extend([
            # nn.AdaptiveAvgPool2d((7, 7)),
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are common amongst all classes.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nl.ElementWiseLinear(4096, num_outputs,mask_init=self.mask_init, mask_scale=self.mask_scale,
                threshold_fn=self.threshold_fn))
            # self.classifiers.append(nn.Linear(4096, num_outputs))

    def set_dataset(self, dataset):  # !!!!!!!!!!!!!!!!!!
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16, self).train(mode)
        # Set the BNs to eval mode so that the running means and averages do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ModifiedVGG16BN(ModifiedVGG16):
    """VGG16 with support for multiple classifiers."""

    def __init__(self, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False):
        super(ModifiedVGG16BN, self).__init__(make_model=False)
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original)

    def make_model(self, mask_init, mask_scale, threshold_fn, original):
        """Creates the model."""
        if original:
            vgg16_bn = models.vgg16_bn(pretrained=True)
            print('Creating model: No mask layers.')
        else:
            # Get the one with masks and pretrained model.
            vgg16_bn = modnets.vgg16_bn(mask_init, mask_scale, threshold_fn)
            vgg16_bn_pretrained = models.vgg16_bn(pretrained=True)
            # Copy weights from the pretrained to the modified model.
            for module, module_pretrained in zip(vgg16_bn.modules(), vgg16_bn_pretrained.modules()):
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    if module.running_mean is not None:
                        module.running_mean.copy_(module_pretrained.running_mean)
                        module.running_var.copy_(module_pretrained.running_var)
            print('Creating model: Mask layers created.')

        self.datasets, self.classifiers = [], nn.ModuleList()

        idx = 6
        for module in vgg16_bn.classifier.children():
            if isinstance(module, (nn.Linear, nl.ElementWiseLinear)):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1
        features = list(vgg16_bn.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are common amongst all classes.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None


class ModifiedResNet(ModifiedVGG16):
    """ResNet-50."""

    def __init__(self, args, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False,init=None):
        super(ModifiedResNet, self).__init__(make_model=False)
        self.args=args
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original,init=init)

    def make_model(self, mask_init, mask_scale, threshold_fn, original,init=None):
        """Creates the model."""
        if original:
            resnet = models.resnet50(pretrained=False)

        
        else:
            # Get the one with masks and pretrained model.
            resnet = modnets.resnet50(mask_init, mask_scale, threshold_fn)        
        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        # self.datasets, self.classifiers = [], nn.ModuleList()
        # # Add the default imagenet classifier.
        # self.datasets.append('imagenet')
        # self.classifiers.append(resnet.fc)
        if original:
            self.classifier = nn.Linear(2048, self.args.num_classes,bias=False)
            print('Creating model: No mask layers.')
        else:
            # model.set_dataset() has to be called explicity, else model won't work.
            self.classifier = nl.ElementWiseLinear(2048, self.args.num_classes, bias=True, mask_init=mask_init, mask_scale=mask_scale,
                        threshold_fn=threshold_fn)#None
        if init is not None:
            v,u=[],[]
            for i in self.shared.modules():
                if 'ElementWise' in str(type(i)) or 'BatchNorm' in str(type(i)):
                    v.append(i)
            for i in self.classifier.modules():
                if 'ElementWise' in str(type(i)) or 'BatchNorm' in str(type(i)):
                    v.append(i)
            for i in init.modules():
                # print(str(type(i)))
                if 'conv' in str(type(i)) or 'linear' in str(type(i)) or 'BatchNorm' in str(type(i)):
                    u.append(i)
            assert len(u)==len(v)
            # resnet_pretrained = init  # models.resnet50(pretrained=True)
            # Copy weights from the pretrained to the modified model.
            for module, module_pretrained in zip(v, u):
                # print(str(type(module)),str(type(module_pretrained)))
                if 'ElementWise' in str(type(module)):
                    # print('initial',str(type(module)))
                    module.weight.data.copy_(module_pretrained.weight.data)
                    # if module.bias:
                    # module.bias.data.copy_(module_pretrained.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    # print('initial',str(type(module)))
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    if module.running_mean is not None:
                        module.running_mean.copy_(module_pretrained.running_mean)
                        module.running_var.copy_(module_pretrained.running_var)
            print('Creating model: Mask layers created.')

    # def add_dataset(self, dataset, num_outputs):
    #     """Adds a new dataset to the classifier."""
    #     if dataset not in self.datasets:
    #         self.datasets.append(dataset)
    #         self.classifiers.append(nn.Linear(2048, num_outputs))


class ModifiedDenseNet(ModifiedVGG16):
    """DenseNet-121."""

    def __init__(self, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False):
        super(ModifiedDenseNet, self).__init__(make_model=False)
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original)

    def make_model(self, mask_init, mask_scale, threshold_fn, original):
        """Creates the model."""
        if original:
            densenet = models.densenet121(pretrained=True)
            print('Creating model: No mask layers.')
        else:
            # Get the one with masks and pretrained model.
            densenet = modnets.densenet121(mask_init, mask_scale, threshold_fn)
            densenet_pretrained = models.densenet121(pretrained=True)
            # Copy weights from the pretrained to the modified model.
            for module, module_pretrained in zip(densenet.modules(), densenet_pretrained.modules()):
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    if module.bias:
                        module.bias.data.copy_(module_pretrained.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    if module.running_mean is not None:
                        module.running_mean.copy_(module_pretrained.running_mean)
                        module.running_var.copy_(module_pretrained.running_var)
            print('Creating model: Mask layers created.')

        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = densenet.features

        # Add the default imagenet classifier. 1000分类
        self.datasets.append('imagenet')
        self.classifiers.append(densenet.classifier)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def forward(self, x):
        features = self.shared(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(1024, num_outputs))


class ResNetDiffInit(ModifiedResNet):
    """ResNet50 with non-ImageNet initialization."""

    def __init__(self, source, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False):
        super(ResNetDiffInit, self).__init__(make_model=False)
        if make_model:
            self.make_model(source, mask_init, mask_scale,
                            threshold_fn, original)

    def make_model(self, source, mask_init, mask_scale, threshold_fn, original):
        """Creates the model."""
        if original:
            resnet = torch.load(source)
            print('Loading model:', source)
        else:
            # Get the one with masks and pretrained model.
            resnet = modnets.resnet50(mask_init, mask_scale, threshold_fn)
            resnet_pretrained = torch.load(source)
            # Copy weights from the pretrained to the modified model.
            for module, module_pretrained in zip(resnet.modules(), resnet_pretrained.modules()):
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    if module.bias:
                        module.bias.data.copy_(module_pretrained.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    if module.running_mean is not None:
                        module.running_mean.copy_(module_pretrained.running_mean)
                        module.running_var.copy_(module_pretrained.running_var)
            print('Creating model: Mask layers created.')

        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        # Add the default classifier.
        if 'places' in source:
            self.datasets.append('places')
        elif 'imagenet' in source:
            self.datasets.append('imagenet')
        if original:
            self.classifiers.append(resnet.fc)
        else:
            self.classifiers.append(resnet_pretrained.fc)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None


class ModifiedMotion(nn.Module):
    def __init__(self, args, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False,init=None):
        super(ModifiedMotion, self).__init__()
        self.args=args
        # self.num_class = args.num_classes
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original,init)

    def make_model(self, mask_init, mask_scale, threshold_fn, original,init):
        """Creates the model."""
        if original:
            pass
            # self.shared = CNNLfw(self.args)  # torchvision
            # print('Creating model: No mask layers.')
        else:
            # Get the one with masks and pretrained model.
            self.shared = nn.Sequential(
                nl.ElementWiseConv2d(
                    1, 6, kernel_size=(1, 5),
                    mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(True),
                
                nl.ElementWiseConv2d(
                    6, 16, kernel_size=(1, 5),
                    mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(True),
                
             )
            self.classifier = nn.Sequential(
                nl.ElementWiseLinear(432, 100, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(True),
                nl.ElementWiseLinear(
                    100, 50, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.ReLU(True),
                nl.ElementWiseLinear(50, self.args.num_classes, mask_init=mask_init, 
                mask_scale=mask_scale, threshold_fn=threshold_fn)
            )
            v,u=[],[]
            for i in self.shared.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in self.classifier.modules():
                if 'ElementWise' in str(type(i)):
                    v.append(i)
            for i in init.modules():
                if 'conv' in str(type(i)) or 'linear' in str(type(i)):
                    u.append(i)
            assert len(u)==len(v)
            for module, module_pretrained in zip(v, u):
                # if isinstance(module, (nl.ElementWiseConv2d, nl.ElementWiseLinear)):
                if 'ElementWise' in str(type(module)):
                    print('initializing',str(type(module)))
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
            # print('Creating model: Mask layers created.',module_pretrained.weight.data)

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedMotion, self).train(mode)
        # Set the BNs to eval mode so that the running means and averages do not update.
        # pass
        # for module in self.shared.modules():
        #     if 'BatchNorm' in str(type(module)):
        #         module.eval()
    def forward(self, x):
        x = self.shared(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.classifier(x)
        return x


class ModifiedConvNet(nn.Module):###for cifar10

    def __init__(self, args, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False, num_class=1000, init=None):
        super(ModifiedConvNet,self).__init__()
        self.args = args
        self.num_class = args.num_classes
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original, init,num_classes=num_class)

    def make_model(self, mask_init, mask_scale, threshold_fn, original, init, width=64, num_classes=10, num_channels=3):
        """Creates the model."""
        if original:
            pass
        else:
            # Get the one with masks and pretrained model.
            self.shared = nn.Sequential(
                nl.ElementWiseConv2d(
                    num_channels, 1 * width, kernel_size=3, padding=1,
                    mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
                nn.BatchNorm2d(1 * width, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True), #inplace 可以载入更大模型

                nl.ElementWiseConv2d(1 * width, 2 * width, kernel_size=3, padding=1,mask_init=mask_init,
                    mask_scale=mask_scale, threshold_fn=threshold_fn),           # output[128, 27, 27]
                nn.BatchNorm2d(2 * width, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True),  #inplace 可以载入更大模型      # output[128, 13, 13]

                nl.ElementWiseConv2d(2 * width, 2 * width, kernel_size=3, padding=1,mask_init=mask_init,
                    mask_scale=mask_scale, threshold_fn=threshold_fn),          # output[192, 13, 13]
                nn.BatchNorm2d(2 * width, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True),  # inplace 可以载入更大模型      # output[128, 13, 13]

                nl.ElementWiseConv2d(2 * width, 4 * width, kernel_size=3, padding=1, mask_init=mask_init,
                                     mask_scale=mask_scale, threshold_fn=threshold_fn),  # output[192, 13, 13]
                nn.BatchNorm2d(4 * width, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True),  # inplace 可以载入更大模型      # output[128, 13, 13]
                #conv4
                nl.ElementWiseConv2d(4 * width, 4 * width, kernel_size=3, padding=1, mask_init=mask_init,
                                     mask_scale=mask_scale, threshold_fn=threshold_fn),  # output[192, 13, 13]
                nn.BatchNorm2d(4 * width, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True),  # inplace 可以载入更大模型      # output[128, 13, 13]

                nl.ElementWiseConv2d(4 * width, 4 * width, kernel_size=3, padding=1, mask_init=mask_init,
                                     mask_scale=mask_scale, threshold_fn=threshold_fn),  # output[192, 13, 13]
                nn.BatchNorm2d(4 * width, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True),  # inplace 可以载入更大模型      # output[128, 13, 13]

                nn.MaxPool2d(kernel_size=3),

                nl.ElementWiseConv2d(4 * width, 4 * width, kernel_size=3, padding=1, mask_init=mask_init,
                                     mask_scale=mask_scale, threshold_fn=threshold_fn),  # output[192, 13, 13]
                nn.BatchNorm2d(4 * width, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True),  # inplace 可以载入更大模型      # output[128, 13, 13]

                nl.ElementWiseConv2d(4 * width, 4 * width, kernel_size=3, padding=1, mask_init=mask_init,
                                     mask_scale=mask_scale, threshold_fn=threshold_fn),  # output[192, 13, 13]
                nn.BatchNorm2d(4 * width, eps=1e-05, momentum=0.1, affine=self.args.affine, track_running_stats=False),
                nn.ReLU(inplace=True),  # inplace 可以载入更大模型      # output[128, 13, 13]

                nn.MaxPool2d(kernel_size=3),
            )
            self.classifier = nn.Sequential(
                nl.ElementWiseLinear(36 * width, num_classes, mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn),
            )
            v, u = [], []
            for i in self.shared.modules():
                if 'ElementWise' in str(type(i)) or 'BatchNorm' in str(type(i)):
                    v.append(i)
            for i in self.classifier.modules():
                if 'ElementWise' in str(type(i)) or 'BatchNorm' in str(type(i)):
                    v.append(i)
            for i in init.modules():
                if 'conv' in str(type(i)) or 'linear' in str(type(i)) or 'batchnorm' in str(type(i)):
                    u.append(i)
            print(u,'\n',v)
            assert len(u) == len(v)
            for module, module_pretrained in zip(v, u):
                # if isinstance(module, (nl.ElementWiseConv2d, nl.ElementWiseLinear)):
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)

                elif 'BatchNorm' in str(type(module)):
                    if module.weight is not None:
                        module.weight.data.copy_(module_pretrained.weight.data)
                        module.bias.data.copy_(module_pretrained.bias.data)
                    if module.running_mean is not None:
                        module.running_mean.copy_(module_pretrained.running_mean)
                        module.running_var.copy_(module_pretrained.running_var)
            print('Creating model: Mask BN layers created.')


    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedConvNet, self).train(mode)
        # Set the BNs to eval mode so that the running means and averages do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()
    def forward(self, x):
        x = self.shared(x)
        x = torch.flatten(x, start_dim=1) #展平   或者view() x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
