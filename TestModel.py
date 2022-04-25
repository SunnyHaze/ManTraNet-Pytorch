import torch
from Train import MyDataset
from imports.ParametersManager import *
from Mantra_Net import *
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

# Enter the *.pt model file name here to load parameters
DIR = './Pre_TrainedModel/'

# ===You need to change the name of the model here =====
ModelName = DIR + 'MantraNet on NIST16_model (8).pt' 
# ====================================================

parManager = ParametersManager('cuda')
parManager.loadFromFile(ModelName)
print("This model has done : {} Epochs.".format(parManager.EpochDone))
model = ManTraNet()
model.cuda()
parManager.setModelParameters(model)

TrainSetDIR = './NIST2016/Train.csv'
TestSetDIR = './NIST2016/Test.csv'

'''
You can set the TrainSetDIR or TestSetDIR to validate on different dataset.
'''
data = MyDataset(TrainSetDIR)

with torch.no_grad():
    model.eval()
    Loader = DataLoader(data, pin_memory=True, batch_size=1, sampler= torch.utils.data.sampler.SubsetRandomSampler(range(len(data))))
    trans = transforms.ToPILImage()
    for (x,label) in Loader:
        out = model(x.cuda())
        x = trans(torch.squeeze(x,0))
        label[0,0,0] = 1
        y = trans(torch.squeeze(label,0))
        z = trans(torch.squeeze(out.cpu(),0))
        q = trans(torch.squeeze((out > 0.5).float().cpu(), 0 ))
        print("mean(output): \n",torch.mean(out.cpu()))
        print("\nmean((out>0.5).float) : \n", torch.mean((out > 0.2).float().cpu()))
        print("\nmean(label):\n" ,torch.mean(label))
        plt.subplot(1,4,1)
        plt.imshow(x, cmap='gray')
        plt.subplot(1,4,2)
        plt.imshow(y, cmap='gray')
        plt.subplot(1,4,3)
        plt.imshow(z, cmap='gray')
        plt.subplot(1,4,4)
        plt.imshow(q, cmap='gray')
        plt.show()
        plt.close()
