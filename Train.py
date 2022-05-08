import os
from matplotlib.pyplot import imshow

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import conv2d, dropout, nn, sigmoid, tensor
from torch.utils.data import random_split
import numpy as np
from imports.ParametersManager import *
from Mantra_Net import *
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import csv

'''
Notice: This Trainning script is used to train on NIST16 manipulation detect dataset.

Spliting rate of Trainning set and Test set is shown below: (You can create your own split by run codes in ./NC2016_Test0613/)

    ---Raw Dataset---
    There are 1124 pictures in total.
    There are 49.82% of No-manipulated pictures while 50.18% of manipulated.
    ---Splited dataset---
    len(Train.csv): 900
    len(test.csv): 224
        ---Trainning set proportion---
        There are 49.89% of No-manipulated pictures while 50.11% of manipulated.
        ---Testing set proportion---
        There are 49.55% of No-manipulated pictures while 50.45% of manipulated.
'''

# 超参数
# Super parameters
MODELNAME='MantraNet on NIST16'  # Name of the model
MODELFILEDIR = './'              # 模型参数存储路径  The saving dir for model parameters
BatchSize = 4
LEARNINGRATE = 1e-5
epochNums = 5
SaveModelEveryNEpoch = 2         # 每执行多少次保存一个模型 Save model when runing every n epoch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 构建模型参数文件存取路径
# Constuct the path for saving and load the model parameters.
if not os.path.exists(MODELFILEDIR):
    os.mkdir(MODELFILEDIR)
MODELFILEPATH = os.path.join(MODELFILEDIR, MODELNAME+'_model.pt')

# 图片素材和索引路径：
# Image file path (Dataset) and the path of indexing csv file 
ImagePath = './NIST2016/'
TrainDatasetIndex = './NIST2016/Train.csv'
TestDatasetIndex = './NIST2016/Test.csv'

# 可以将数据线包装为Dataset，然后传入DataLoader中取样
# Build a Dataset for local datas
class MyDataset(Dataset):
    def __init__(self, Path) -> None:
        with open(Path, 'r') as f:
            reader = csv.reader(f)
            self.index = []
            for i in reader:
                self.index.append(i)   
        self.trans = transforms.ToTensor()  
     
    def __getitem__(self, i):
        image = Image.open("{}{}".format(ImagePath, self.index[i][0]))
        image = self.trans(image)
        if self.index[i][1] != 'N':
            '''
            this part is to generate a mask for manipulated images
            '''
            mask = Image.open("{}{}".format(ImagePath, self.index[i][1]))
            mask = mask.convert("1") # convert to 0-1 image with PIL api
            mask = self.trans(mask)
        else:
            ''' 
            torch.ones(...) generates a totally white image which represent to a mask of NO manipulation 
            '''
            mask = torch.ones((1, image.shape[1], image.shape[2]))
        return image, mask
    
    def __len__(self):
        return len(self.index)

# 定义准确率函数
# defination of accracy function
def accuracy(output:torch.Tensor , mask):
    output = (output > 0.5).float()
    error = torch.sum(torch.abs(output - mask))
    acc = 1 - error / (BatchSize * mask.shape[2] * mask.shape[3])
    return acc

if __name__ == "__main__":    
    # 模型实例化
    # Instantiation of the model
    model = ManTraNet()
    model.cuda()
    # print(model)
    
    # 如果有“半成品”则导入参数
    # If there is a pre-trained model, load it.
    parManager = ParametersManager(device)
    if os.path.exists(MODELFILEPATH):
        parManager.loadFromFile(MODELFILEPATH)
        parManager.setModelParameters(model)
    else:
        print('===No pre-trained model found!===')

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNINGRATE)
    
    # 构建数据集
    # Constrct the dataset
    TrainDataset = MyDataset(TrainDatasetIndex)
    TestDataset = MyDataset(TestDatasetIndex)

    print('Trainset size: {}'.format(len(TrainDataset)))
    print('Testset size: {}'.format(len(TestDataset)))

    # # 分割数据集
    # # Split dataset in TrainingSet and TestSet
    # TrainDataset, TestDataset = random_split(dataset = Datas, lengths = [train_size,test_size],generator=torch.Generator().manual_seed(0))

    # 构建训练集读取器
    # Consruct the Dataloader for both datasets
    TrainLoader = DataLoader(TrainDataset,num_workers=8, pin_memory=True, batch_size=BatchSize, sampler= torch.utils.data.sampler.SubsetRandomSampler(range(len(TrainDataset))))
    # 构建测试集读取器：
    TestLoader = DataLoader(TestDataset,num_workers=8, pin_memory=True, batch_size=BatchSize, sampler= torch.utils.data.sampler.SubsetRandomSampler(range(len(TestDataset))))
    
    # 输出训练集长度 print the length of training set
    print('len(TrainLoader):{}'.format(len(TrainLoader)))
    
    TrainACC = []
    TestACC = []
    GlobalLoss = []
    for epoch in range(epochNums):
        print("===开始本轮的Epoch {} == 总计是Epoch {}===".format(epoch, parManager.EpochDone))
        
        # 收集训练参数
        # Collect the tranning statistics
        epochAccuracy = []
        epochLoss = []
        model.train()
        #=============实际训练流程=================
        #=============Trainning step start=================
        for batch_id, (inputs,label) in enumerate(TrainLoader):
            # torch.train()
            optimizer.zero_grad()
            output = model(inputs.cuda())          
            loss = criterion(output,label.cuda())
            loss.backward()
            optimizer.step()
            epochAccuracy.append(accuracy(output,label.cuda()).cpu())
            epochLoss.append(loss.item())
            # print status
            if batch_id % (int(len(TrainLoader) / 20)) == 0: 
                print("    Now processing step[{}/{}], Current Epoch accuracy：{:.2f}%，Loss：{:.8f}".format(batch_id,len(TrainLoader), np.mean(epochAccuracy) * 100, loss))
        #==============本轮训练结束==============
        #=============Trainning step finish=================
        # 收集训练集准确率
        TrainACC.append(np.mean(epochAccuracy)) 
        GlobalLoss.append(np.mean(epochLoss))
        # ==========进行一次验证集测试============
        # ==========Start a test set test============
        localTestACC = []
        model.eval() # 进入评估模式，节约开销
        for inputs, label in TestLoader:
            torch.no_grad() # 上下文管理器，此部分内不会追踪梯度/
            output = model(inputs.cuda())
            localTestACC.append(accuracy(output,label.cuda()).cpu())
        # ==========验证集测试结束================
        # ==========test set test done============
        TestACC.append(np.mean(localTestACC))
        print("Current Epoch Done, Train accuracy: {:3f}%, Test accuracy: {:3f}%".format(TrainACC[-1] * 100, TestACC[-1] * 100))
        # 暂存结果到参数管理器
        # Save results to parameters-manager
        parManager.oneEpochDone(LEARNINGRATE,TrainACC[-1],TestACC[-1],GlobalLoss[-1])
        # 周期性保存结果到文件
        # Save model to file periodically
        if epoch == epochNums - 1 or epoch % SaveModelEveryNEpoch == 0:
            parManager.loadModelParameters(model)
            parManager.saveToFile(MODELFILEPATH)
            
    # ===========view the results=============
    parManager.show()
    plt.figure(figsize=(10,7))
    plt.subplots_adjust(left=0.1,bottom=0.1,top=0.9,right=0.9,wspace=0.1,hspace=0.3)
    plt.subplot(2,1,1)
    plt.plot(range(parManager.EpochDone),parManager.TrainACC,marker='*' ,color='r',label='Train')
    plt.plot(range(parManager.EpochDone),parManager.TestACC,marker='*' ,color='b',label='Test')

    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend()
    plt.title("{} on Nist".format(MODELNAME))
    plt.text(int(parManager.EpochDone *0.8),0.5,'Train ACC: {:.6f}\nTest ACC: {:.6f}\nEpoch:{}'.format(parManager.TrainACC[-1],parManager.TestACC[-1], parManager.EpochDone))
    plt.subplot(2,1,2)
    plt.title('Learning Rates')
    plt.xlabel('Epoch')
    plt.ylabel('$log_{10}$(Learning Rates)')
    plt.ylim(0,-5)
    plt.plot([x for x in range(parManager.EpochDone)], np.log(parManager.LearningRate) / np.log(10))
    plt.savefig('Train-{}-{}Epoch.jpg'.format(MODELNAME,parManager.EpochDone))
    plt.show()