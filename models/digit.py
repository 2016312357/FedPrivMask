#
# CODE COURTESY: https://github.com/med-air/FedBN/blob/master/nets/models.py
#
# WHEN USING INPUT OF MNIST AND KMNIST, WE USE THIS MODEL
#

import torch.nn as nn
import torch.nn.functional as func

class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10,in_channels=3, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*8*8, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)
        self.feature=None
    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)
        x = self.fc2(x)
        x = self.bn5(x)
        self.feature=func.relu(x)
        x = self.fc3(self.feature)
        return x
    def extract_feature(self):
        return self.feature  #######last output

class MoonDigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits.
    """

    def __init__(self, num_classes=10, in_channels=3, **kwargs):
        super(MoonDigitModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 8 * 8, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        h = func.relu(x)
        x = self.fc3(h)
        return h, x