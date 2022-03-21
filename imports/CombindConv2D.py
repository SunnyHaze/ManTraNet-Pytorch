import os
from turtle import forward
import torch
from torch import nn, tensor
import torch.nn.functional
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
'''
    In my Own under standing of the SRM layer, with input chanel 3 and output chanle3, but different with the Mantra-Net source code.
'''
# class SRMConv2D(nn.Module):
#     def __init__(self):
#         super(SRMConv2D,self).__init__()
#         q = [4, 12, 2] # coefficient of the kernels
#         self.kernel1 = np.array([
#             [0, 0, 0, 0, 0],
#             [0,-1, 2,-1, 0],
#             [0, 2,-4, 2, 0],
#             [0,-1, 2,-1, 0],
#             [0, 0, 0, 0, 0]
#         ],dtype=np.float32)
#         self.kernel2 = np.array([
#             [-1, 2,-2, 2,-1],
#             [2, -6, 8,-6, 2],
#             [-2, 8,-12,8,-2],
#             [2, -6, 8,-6, 2],
#             [-1, 2,-2, 2,-1]          
#         ],dtype=np.float32)
#         self.kernel3 = np.array([
#             [0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#             [0, 1,-2, 1, 0],
#             [0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#         ],dtype=np.float32)
#         # shape (3,3,5,5)
#         weight = torch.tensor( np.array([ 
#             [self.kernel1 / q[0] for i in range(3)],
#             [self.kernel2 / q[1] for i in range(3)],
#             [self.kernel3 / q[2] for i in range(3)],
#         ]),dtype=torch.float32)
#         print(weight)
#         # weight = torch.transpose(weight)
#         self.weight = torch.nn.Parameter(weight, requires_grad=False) 

#     def forward(self, x):
#         with torch.no_grad():
#             return torch.nn.functional.conv2d(x, weight=self.weight, padding = 2)
'''
  BayarConv2D, refering from 'Constrained Convolutional Neural Networks: A New Approach Towards General Purpose Image Manipulation Detection'
'''
class BayarConv2D(nn.Module):
    def __init__(self ,inputchanel, outputchanel, kernelsize) :
        super(BayarConv2D,self).__init__()
        self.mask = None
        weight = torch.Tensor(inputchanel, outputchanel, kernelsize, kernelsize)
        self.weight = torch.nn.Parameter(weight)
        nn.init.xavier_normal_(self.weight)
        # print(self.weight)
    
    def _initialize_mask(self) :
        chanelin = self.weight.shape[0]
        chanelout  = self.weight.shape[1]
        ksize = self.weight.shape[2] 
        m = np.zeros([chanelin, chanelout, ksize, ksize]).astype('float32')
        m[:,:,ksize//2,ksize//2] = 1.
        self.mask = torch.tensor(m).cuda()
    
    def _get_new_weight(self) :
        with torch.no_grad():
            if self.mask is None :
                self._initialize_mask()
            self.weight.data *= (1-self.mask)
            # print(self.weight)
            rest_sum = torch.sum(self.weight, dim=(2,3), keepdims=True)
            # print('sum')
            # print(rest_sum)
            # print(rest_sum.shape)
            self.weight.data /= rest_sum + 1e-7
            self.weight.data -= self.mask
            # print(self.weight)
            # print(self.weight.grad)
    
    def forward(self, x):
        self._get_new_weight()
        return torch.nn.functional.conv2d(x, weight=self.weight, padding = 2)

''' 
    Kernel coefficient copy from the Mantra-Net source code, with 3 input chanels and 9 output chanels, which is unexpected different from the papar.
'''
class SRMConv2D(nn.Module):
    def _get_srm_list(self) :
        # srm kernel 1                                                                                                                                
        srm1 = np.zeros([5,5]).astype('float32')
        srm1[1:-1,1:-1] = np.array([[-1, 2, -1],
                                    [2, -4, 2],
                                    [-1, 2, -1]] )
        srm1 /= 4.
        # srm kernel 2                                                                                                                                
        srm2 = np.array([[-1, 2, -2, 2, -1],
                         [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2],
                         [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]]).astype('float32')
        srm2 /= 12.
        # srm kernel 3                                                                                                                                
        srm3 = np.zeros([5,5]).astype('float32')
        srm3[2,1:-1] = np.array([1,-2,1])
        srm3 /= 2.
        return [ srm1, srm2, srm3 ]
    
    def _build_SRM_kernel(self) :
        kernel = []
        srm_list = self._get_srm_list()
        for idx, srm in enumerate( srm_list ):
            for ch in range(3) :
                this_ch_kernel = np.zeros([5,5,3]).astype('float32')
                this_ch_kernel[:,:,ch] = srm
                kernel.append( this_ch_kernel )
        kernel = np.stack( kernel, axis=-1 )
        # srm_kernel = K.variable( kernel, dtype='float32', name='srm' )
        '''
        Keras kernel form   (kernel_width, kernel_height, inputChanels, outputChanels)
        pytorch Kernal form (inputChanels, outputChanel, kernel_size, kernel_size)
        
        There is a need to switch the dim to fit in pytorch with the Mantra-Net source code writting in keras.
        '''
        kernel = np.swapaxes(kernel,1,2)
        # kernel = np.swapaxes(kernel,1,2)
        kernel = np.swapaxes(kernel,0,3)      
        return kernel
    
    def __init__(self):
        super(SRMConv2D,self).__init__()
        self.weight = torch.tensor(self._build_SRM_kernel()).cuda()
    def forward(self, x):
        with torch.no_grad():
            return torch.nn.functional.conv2d(x, weight=self.weight, padding = 2)

class CombindConv2D(nn.Module):
    def __init__(self, outputChanels) -> None:
        super(CombindConv2D, self).__init__()
        self.subLayer1 = BayarConv2D(3,3,5) # outchanel 3
        self.relu1 = nn.ReLU(inplace=True)
        self.subLayer2 = SRMConv2D()        # outchanel 9 
        self.relu2 = nn.ReLU(inplace=True)
        self.subLayer3 = nn.Conv2d(3,outputChanels - 3 - 9, kernel_size=5, padding=2) # 总数-12个普通卷积层
        self.relu3 = nn.ReLU(inplace=True)
    def forward(self,x):
        x1 = self.subLayer1(x)
        x1 = self.relu1(x1)
        x2 = self.subLayer2(x)
        x2 = self.relu2(x2)
        x3 = self.subLayer3(x)
        x3 = self.relu3(x3)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        x = torch.cat([x1,x2,x3], dim=1)
        # print(x.shape)
        return x

if __name__ =='__main__':
    # 测试完整Combind
    # net = CombindConv2D(16)
    # # image_dir = 'NC2016_Test0613/world/NC2016_2198.jpg'
    # image_dir = 'NC2016_Test0613/probe/NC2016_8411.jpg'
    # # image_dir = '1.jpg'
    # image = Image.open(image_dir)
    # image = np.array(image)
    # print(image.shape)
    # trans = transforms.ToTensor()
    # t = trans(image).unsqueeze(0)
    # t = net(t)
    # print(net)
    # print(t.shape)
    ##########
    # 测试
    # a = torch.Tensor()
    # b = BayarConv2D(3,3,5)
    # b._initialize_mask()
    # b._get_new_weight()
    net = SRMConv2D()
    # image_dir = 'NC2016_Test0613/world/NC2016_2198.jpg'
    image_dir = 'NC2016_Test0613/probe/NC2016_8411.jpg'
    # image_dir = 'NC2016_Test0613/probe/NC2016_6003.jpg'
    # image_dir = '1.jpg'
    image = Image.open(image_dir).resize((1200,1200))
    image = np.array(image)
    # image = np.concatenate([image,image,image],axis=-1)
    print(image.shape)
    trans = transforms.ToTensor()
    t = trans(image).unsqueeze(0)
    t = net(t)
    print(t[0].shape)
    # 3层直接输出
    trans = transforms.ToPILImage()
    # img = trans(t[0][3:9])
    # ########## 超厚
    # print(t[0] * 255)
    # number = np.array(img)
    # print(np.max(number))
    # print(np.min(number))
    final = torch.zeros(3,image.shape[0],image.shape[1])
    final[0] += t[0][0]
    final[0] += t[0][3]
    final[0] += t[0][6]
    final[1] += t[0][1]
    final[1] += t[0][4]
    final[1] += t[0][7]
    final[2] += t[0][2]
    final[2] += t[0][5]
    final[2] += t[0][8]
    print(final)
    img=trans(final)
    # 单层输出
    t = t[0][0]
    M = torch.max(final)
    Mi = torch.min(final)
    print(M,' ',Mi)
    # img = np.array((final- Mi)/(M - Mi)  * 255, dtype=np.float32)
    img = trans(final)
    print(np.max(img))
    print(np.min(img))
    plt.imshow(img, cmap='jet')
    plt.show()