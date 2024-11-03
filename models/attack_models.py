# Python version: 3.6
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
different attack model architectures
'''


class AttackNet(nn.Module):
    def __init__(self, in_dim=337721, out_dim=1):
        super(AttackNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __str__(self):
        return 'attack'


class AttackNet1(nn.Module):
    def __init__(self, in_dim=337721, out_dim=1):
        super(AttackNet1, self).__init__()
        self.fc1 = nn.Linear(in_dim, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __str__(self):
        return 'attack-1'


class AttackNet2(nn.Module):
    def __init__(self, in_dim=337721, out_dim=1):
        super(AttackNet2, self).__init__()
        self.fc1 = nn.Linear(in_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __str__(self):
        return 'attack-2'


class AttackNet4(nn.Module):
    def __init__(self, in_dim=10, out_dim=1):
        super(AttackNet4, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        # self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(64, out_dim)
        # self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def __str__(self):
        return 'attack-4'
