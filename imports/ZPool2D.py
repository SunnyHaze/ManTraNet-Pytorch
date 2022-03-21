import torch
from torch import nn, tensor
import torch.nn.functional
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class Zpool2D_Window(nn.Module):
    def __init__(self, inputChannels, window_size_list, min_value=1e-5) -> None:
        super().__init__()
        self.min_value = min_value
        self.window_size_list = window_size_list
        self.maxWinSize = np.max(window_size_list)
        
        self.TinyWeight  = nn.Parameter(torch.full([1,1,inputChannels,1,1], min_value, dtype=torch.float32) ,requires_grad= True)
        self.TinyWeight.data.clamp(min=0)

    def _init_padding_buff(self, x): # include Cumulate sum
        paddingLayer = nn.ZeroPad2d(self.maxWinSize//2 + 1)
        x_pad = paddingLayer(x)
        x_cum = torch.cumsum(x_pad, 2)
        x_cum = torch.cumsum(x_cum, 3)
        return x_cum
                
    def _compute_a_window_avg(self, x, winSize):
        # --left top Big square block-- coordinate
        top = self.maxWinSize // 2 - winSize // 2
        bottom = top + winSize
        left = self.maxWinSize // 2 - winSize // 2
        right = left + winSize 

        Ax, Ay = (left, top)
        Bx, By = (right, top)
        Cx, Cy = (right, bottom)
        Dx, Dy = (left, bottom)
         
        # negative number , but can be parse to a positve when using fomula like this -> [:-1]
        
        # --right bottom Big square block-- coordinate
        top0 = -self.maxWinSize // 2 - winSize // 2 - 1
        bottom0 = top0 + winSize 
        left0 = -self.maxWinSize // 2 - winSize // 2 - 1 
        right0 = left0 + winSize
        
        Ax0, Ay0 = (left0, top0)
        Bx0, By0 = (right0, top0)
        Cx0, Cy0 = (right0, bottom0)
        Dx0, Dy0 = (left0, bottom0)
        
        counts = torch.ones_like(x)
        # print(counts)
        counts_pading = self._init_padding_buff(counts)
        # print(counts_pading)
        x_padding = self._init_padding_buff(x)

        counts_2d = counts_pading[:,:,Ay:Ay0, Ax:Ax0] \
                  + counts_pading[:,:,Cy:Cy0, Cx:Cx0] \
                  - counts_pading[:,:,By:By0, Bx:Bx0] \
                  - counts_pading[:,:,Dy:Dy0, Dx:Dx0]

        sum_x_2d = x_padding[:,:,Ay:Ay0, Ax:Ax0] \
                 + x_padding[:,:,Cy:Cy0, Cx:Cx0] \
                 - x_padding[:,:,By:By0, Bx:Bx0] \
                 - x_padding[:,:,Dy:Dy0, Dx:Dx0]
        avg_x_2d = sum_x_2d / counts_2d
        return avg_x_2d
    
    def forward(self, x):
        outputFeature = []
        # 1. window
        for win in self.window_size_list:
            avg_x_2d = self._compute_a_window_avg(x, win)
            D_x = x - avg_x_2d
            outputFeature.append(D_x)
        # 2. global
        mu_f = torch.mean(x, dim=(2,3), keepdim=True)
        D_f = x - mu_f
        outputFeature.append(D_f)
        # 5 Dim Tensor arrange : (Batch, Diff_Windows, channel, width, height )
        outputFeature = torch.stack(outputFeature,1) 
        
        std_x = torch.std(outputFeature, dim=(3,4),keepdim=True)
        std_x = torch.maximum(std_x, self.TinyWeight + self.min_value / 10.)
        
        x = torch.stack([x for i in range(len(self.window_size_list)+ 1) ], dim=1)
        Z_f = x / std_x

        return Z_f
# a = np.ones(60) * 1.1
a = np.arange(0,36 * 3,1)
# a = np.zeros(60)
a = np.resize(a, (1,3,6,6))
a = torch.tensor(a, dtype=torch.float32)
# print(a)
net = Zpool2D_Window(3, [3,5,7])
# a = net._init_padding_buff(a)
# print(a)
a = net(a)
# print(net._init_padding_buff(a))