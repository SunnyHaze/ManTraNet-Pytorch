import os
from idna import valid_string_length
from matplotlib.pyplot import imshow
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import conv2d, dropout, nn, sigmoid, tensor
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
# Layer files
from imports.ParametersManager import * # import training-help tools
from imports.CombindConv2D import *         # import defination of special layers
from imports.ZPool2D import *           # inport Z-Pooling layers
from imports.convlstm import * # Copied from https://github.com/ndrplz/ConvLSTM_pytorch

# Hyperparameter
ZPoolingWindows = [7, 15, 31]

# L2Norm Layer
class L2Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        a = torch.norm(x,2, keepdim=True) # 对于整个通道层求L2范数，并利用其进行标准化
        x = x / a
        return x
    
class ManTraNet(nn.Module):
    def __init__(self) -> None:
        super(ManTraNet, self).__init__()
        self.combind = CombindConv2D(32) # 此处填入数值n - 9（SRM） - 3（Bayer） 后是实际存在的卷积层个数
        self.vgg = nn.Sequential( # 全连接Conv2D，没有pooling 
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            L2Norm()
        )
        # initialize the Conv layer with xavier_normal_
        for m in self.vgg.modules():
            if isinstance(m, nn.Linear):
                pass
            # 也可以判断是否为conv2d，使用相应的初始化方式 
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight) 
                
        self.Adaption = nn.Conv2d(256, 64, 1, 1, 0)
        self.BN = nn.BatchNorm2d(64, momentum=0.99, eps=0.001)
        self.ZPool = Zpool2D_Window(64, ZPoolingWindows)
        self.ConvLstm2D = ConvLSTM(input_dim = 64, hidden_dim=8, kernel_size=(7,7), num_layers=1, batch_first=True)
        # todo 
            # ConvLSTM 8@7x7
        self.decision = nn.Conv2d(8, 1, 7, 1, 3)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        # Image H x W x 3
        x = self.combind(x)
        x = self.vgg(x)
        # Feat H x W x 256
        x = self.Adaption(x)
        x = self.BN(x)
        x = self.ZPool(x)
        _, last_states = self.ConvLstm2D(x)
        x = last_states[0][0] # Todo 这里小概率有问题
        x = self.decision(x)
        x = self.sig(x)
        return x

if __name__ == "__main__":
    a = torch.tensor(np.arange(0, 6 * 64 *64, 1).reshape(2,3,64,64), dtype=torch.float32)
    net = ManTraNet()
    print(net(a).shape)
    # a = torch.tensor(np.arange(0,60,1).reshape((5,3,2,2)), dtype=torch.float32 )
    # layer = L2Norm()
    # print(a.shape)
    # a = layer(a)
    # print(a)
    
