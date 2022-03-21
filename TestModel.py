from operator import mod
import torch
from Train import MyDataset
from imports.ParametersManager import *
from Mantra_Net import *
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

# Enter the *.pt model file name here to load parameters
ModelName = './MantraNet on NIST16_model (8).pt' 
ModelName = './MantraNet on NIST16_model (13).pt'

parManager = ParametersManager('cuda')
parManager.loadFromFile(ModelName)

model = ManTraNet()
model.cuda()
parManager.setModelParameters(model)

data = MyDataset()

with torch.no_grad():
    model.eval()
    Loader = DataLoader(data, pin_memory=True, batch_size=1, sampler= torch.utils.data.sampler.SubsetRandomSampler(range(len(data))))
    trans = transforms.ToPILImage()
    for (x,label) in Loader:
        print(x.shape)
        print(label.shape)
        out = model(x.cuda())
        print(out.shape)

        x = trans(torch.squeeze(x,0))

        y = trans(torch.squeeze(label,0))
        z = trans(torch.squeeze(out.cpu(),0))
        q = trans(torch.squeeze((out > 0.5).float().cpu(), 0 ))
        print(torch.mean(out.cpu()))
        print(torch.mean((out > 0.5).float().cpu()))
        print(torch.mean(label))
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
