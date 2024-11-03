#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math

import torch
from torch import nn
import torch.nn.functional as F


class CNNMotion(nn.Module):
    def __init__(self, args):
        super(CNNMotion, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(1, 5))  # in out
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(1, 5))
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(432, 100)  # 16 * 41 * 51,
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, args.num_classes)
        self.feature = None
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        self.feature = F.relu(self.fc2(x))
        # self.feature = x
        x = self.fc3(self.feature)
        return x

    def extract_feature(self):
        return self.feature  #######last output


class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, ),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, ),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, ),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, ),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=args.affine, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, ),

        )

        self.classifier = nn.Sequential(
            nn.Linear(args.fc_input, 4096, bias=True),  # args.fc_input=15360
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(4096, args.num_classes, bias=True)
            # nn.Sigmoid()
        )
        self.feature = None

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        # out = self.classifier(x)
        for i, layer in enumerate(list(self.classifier)[:-1]):
            x = layer(x)
        self.feature = x
        out = list(self.classifier)[-1](x)
        return out

    def extract_feature(self):
        return self.feature  #######last output

    def train_nobn(self):
        pass


# FL global model 结构定义
class CNNLfw(nn.Module):  # variant of lenet for CelebA/lfw dataset
    def __init__(self, args):
        super(CNNLfw, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # in out
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2704, 120)  # 16 * 41 * 51,
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)
        self.feature = None

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        self.feature = F.relu(x)
        x = self.fc3(self.feature)
        return x

    def extract_feature(self):
        return self.feature  #######last output


class VGG11(nn.Module):  # the VGG11 architecture
    def __init__(self, args, in_channels=3):
        super(VGG11, self).__init__()
        self.in_channels = 3  # in_channels
        self.num_classes = args.num_classes
        # convolutional layers
        self.shared = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # fully connected linear layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),  # 512 * 5 * 6
            nn.ReLU(),
            # nn.Dropout2d(0.5),
            # nn.Linear(in_features=4096, out_features=4096),
            # nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.num_classes)
            # nn.Dropout2d(0.5),

        )
        # self.classifier = nn.Linear(in_features=4096, out_features=self.num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.shared(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        # x = self.linear_layers(x)
        x = self.classifier(x)
        # if self.num_classes == 1:
        #     out = self.sigmoid(out)
        return x

    def train_nobn(self):
        pass


class LSTM(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, vocab_size, output_dim, batch_size):
        super(LSTM, self).__init__()

        self.batch_size = batch_size

        self.hidden_dim = hidden_dim

        self.word_embeddings = embedding

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=2,
                            dropout=0.5,
                            batch_first=True)

        # The linear layer that maps from hidden state space to output space
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, sentence):
        # sentence = sentence.type(torch.LongTensor)
        # print ('Shape of sentence is:', sentence.shape)

        sentence = sentence.to(device)

        embeds = self.word_embeddings(sentence)
        # print ('Embedding layer output shape', embeds.shape)

        # initializing the hidden state to 0
        # hidden=None

        h0 = torch.zeros(2, sentence.size(0), hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(2, sentence.size(0), hidden_dim).requires_grad_().to(device)

        lstm_out, h = self.lstm(embeds, (h0, c0))
        # get info from last timestep only
        lstm_out = lstm_out[:, -1, :]
        # print ('LSTM layer output shape', lstm_out.shape)
        # print ('LSTM layer output ', lstm_out)

        # Dropout
        lstm_out = F.dropout(lstm_out, 0.5)

        fc_out = self.fc(lstm_out)
        # print ('FC layer output shape', fc_out.shape)
        # print ('FC layer output ', fc_out)

        out = fc_out
        out = F.softmax(out, dim=1)
        # print ('Output layer output shape', out.shape)
        # print ('Output layer output ', out)
        return out


class fc1(nn.Module):  # 全连接网络
    def __init__(self, dim_in, num_classes):
        super(fc1, self).__init__()
        self.feature = None
        self.shared = nn.Sequential(
            nn.Linear(in_features=dim_in, out_features=300),
            nn.ReLU(True),
            nn.Linear(in_features=300, out_features=100),
            nn.ReLU(True),
        )
        self.classifier = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x):
        self.feature = self.shared(x)
        x = self.classifier(self.feature)
        return x

    def extract_feature(self):
        return self.feature  #######last output
    def train_nobn(self):
        pass

class CNNMnist(nn.Module):  # for mnist dataset, 输入单通道图像数据集
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)  # 两个池化，所以是7*7而不是14*14
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, args.num_classes)
        self.feature=None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 将数据平整为一维的
        x = F.relu(self.fc1(x))
        self.feature = F.relu(self.fc2(x))
        x = self.fc3(self.feature)
        # x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
        return x
    def train_nobn(self):
        pass
    def extract_feature(self):
        return self.feature  #######last output

class Classifier(nn.Module):
    def __init__(self, nc, ndf, nz, release=None):
        super(Classifier, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        self.release = release
        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def forward(self, x, release=None):
        x = x.view(-1, self.nc, 64, 64)
        x = self.encoder(x)
        x = x.view(-1, self.ndf * 8 * 4 * 4)
        x = self.fc(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, args=None, init_weights=True):
        super(AlexNet, self).__init__()
        if args is None:
            self.fc_input = 2560
        else:
            self.fc_input = args.fc_input
        self.shared = nn.Sequential(  # 打包
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55] 自动舍去小数点后
            nn.ReLU(inplace=True),  # inplace 可以载入更大模型
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27] kernel_num为原论文一半
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # 全链接
            nn.Linear(self.fc_input, 2048),  # 128 * 6 * 6
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        # if init_weights:
        #     self._initialize_weights()

    def forward(self, x):
        x = self.shared(x)
        x = torch.flatten(x, start_dim=1)  # 展平   或者view()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                nn.init.constant_(m.bias, 0)

class PurchaseClassifier(nn.Module):  # for purchase100 dataset
    def __init__(self, num_classes=100):
        super(PurchaseClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(600, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        hidden_out = self.features(x)
        out = self.classifier(hidden_out)
        return out


class BasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_branch(x) +
                                     self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      stride=stride,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BottleNeck.expansion,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BottleNeck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_branch(x) +
                                     self.shortcut(x))


class ResNet(nn.Module):  # define Resnet structure
    def __init__(self, block, layers, num_classes=100, inter_layer=False):
        super(ResNet, self).__init__()
        self.inter_layer = inter_layer
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.stage2 = self._make_layer(block, 64, layers[0], 1)
        self.stage3 = self._make_layer(block, 128, layers[1], 2)
        self.stage4 = self._make_layer(block, 256, layers[2], 2)
        self.stage5 = self._make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.feature=None

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.inter_layer:
            x1 = self.stage2(x)
            x2 = self.stage3(x1)
            x3 = self.stage4(x2)
            x4 = self.stage5(x3)
            x = self.avg_pool(x4)
            self.feature = x.view(x.size(0), -1)
            x = self.fc(self.feature)
            return [x1, x2, x3, x4, x]
        else:
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
            x = self.avg_pool(x)
            self.feature = x.view(x.size(0), -1)
            x = self.fc(self.feature)
            return x
    def extract_feature(self):
        return self.feature   


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # only load state_dict()
    # if pretrained:
    #     model.load_state_dict(
    #         torch.load(model_urls[arch], map_location=torch.device('cpu')))
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], pretrained,
                   progress, **kwargs)
