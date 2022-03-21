import torch
from torch import nn
# 定义一个管理模型训练时参数的类
class ParametersManager():
    def __init__(self,device) -> None:
        self.device = device
        # 具体数据
        self.EpochDone = 0      # 已经完成的Epoch个数
        self.LearningRate = []    # 各个Epoch的学习率 
        self.TrainACC = []        # 训练集准确率
        self.TestACC = []         # 测试集准确率
        self.loss = []            # loss
        self.state_dict = 0 # 模型的具体权重
        self.datas = {}
    # 打包
    def pack(self):
        self.datas = {
            'EpochDone' : self.EpochDone,        # 已经完成的Epoch个数
            'LearningRate' : self.LearningRate,    # 各个Epoch的学习率 
            'TrainACC' : self.TrainACC,        # 训练集准确率
            'TestACC' : self.TestACC,         # 测试集准确率
            'loss' : self.loss,            # loss
            'state_dict' : self.state_dict, # 模型的具体权重
        } 
    # 解包
    def unpack(self):
        self.EpochDone = self.datas['EpochDone']
        self.LearningRate = self.datas['LearningRate']
        self.TestACC = self.datas['TestACC']
        self.TrainACC = self.datas['TrainACC']
        self.loss = self.datas['loss']
        self.state_dict = self.datas['state_dict']
    # 从脚本中获取模型的参数
    def loadModelParameters(self, model:nn.Module):
        self.state_dict = model.state_dict()
    
    # 从脚本中将参数输出给模型
    def setModelParameters(self, model:nn.Module):
        model.load_state_dict(self.state_dict)
    
    # 从脚本中获取一个Epoch的
    def oneEpochDone(self, LastLearningRate, LastTrainACC, lastTestACC, lastLoss):
        self.EpochDone += 1
        self.LearningRate.append(LastLearningRate)
        self.TrainACC.append(LastTrainACC)
        self.TestACC.append(lastTestACC)
        self.loss.append(lastLoss)
    
    # 保存数据到文件
    def saveToFile(self, path):
        self.pack()
        torch.save(self.datas, path)
        print('===succesfully saved model!===')
    
    # 从文件中读取数据
    def loadFromFile(self, path):
        self.datas = torch.load(path,map_location=torch.device(self.device))
        self.unpack()
        print('===Load model succesfully!===')
    # 展示当前存储的模型的数据
    def show(self):
        print('===' * 10 + 
'''\n此模型已经训练了{}个Epoch \n
目前的训练集准确率为 {:.3f}% \n
目前的测试集准确率为 {:.3f}% \n'''.format(self.EpochDone, self.TrainACC[-1] * 100, self.TestACC[-1] * 100),'===' * 10)