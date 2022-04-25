import torch
from Train import MyDataset
from imports.ParametersManager import *
from Mantra_Net import *
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
'''
This python file is used to calculate the model' s ROC and AUC value of your trained model.
'''

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
# ===========hyper parameters=============
dataSetChoosen = 'Train'
resolution = 100

# ========================================

dataDIR ={   
'Whole' : './NIST2016/index.csv',
'Train' : './NIST2016/Train.csv',
'Test' : './NIST2016/Test.csv'
}

data = MyDataset(dataDIR[dataSetChoosen])

with torch.no_grad():
    model.eval()
    Loader = DataLoader(data, pin_memory=True, batch_size=1)
    step = int(len(Loader) / 100)
    print(step)
    trans = transforms.ToPILImage()
    labels = []
    prediction = []
    for id, (x,label) in enumerate(Loader):
       labels.append(torch.squeeze(torch.squeeze(label , dim=0), dim=0) )
       out = model(x.cuda())
       prediction.append(torch.squeeze(torch.squeeze(out.cpu(), dim=0), dim=0) )
       if id % step == 0:
           print('{:.2f}%'.format(id/len(Loader) * 100))

    labels = torch.stack(labels, dim = 0)
    prediction = torch.stack(prediction, dim=0)
    
    print(labels.shape)
    print(prediction.shape)
    
    def cal_ROC_rate(labels, predict:torch.Tensor, threshold:float):
        mask = (predict > threshold).float()
        TP, TN, FP, FN = 0, 0, 0, 0
        TP += torch.sum((mask == 1) & (labels == 1))
        TN += torch.sum((mask == 0) & (labels == 0))
        FP += torch.sum((mask == 1) & (labels == 0))
        FN += torch.sum((mask == 0) & (labels == 1))
        TPR = TP / (TP + FN) # True positive rate
        FPR = FP / (TN + FP) # False Positive Rate
        return TPR, FPR

    TPR = []
    FPR = []
    for threshold in range(resolution):
        threshold /= resolution
        # print(threshold)
        t_TPR, t_FPR = cal_ROC_rate(labels, prediction, threshold)
        TPR.append(t_TPR.cpu())
        FPR.append(t_FPR.cpu())
            
    TPR_array = sorted(TPR)
    FPR_array = sorted(FPR)
    
    AUC = np.trapz(TPR_array, FPR_array)   
     
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title('ROC of {} Epoch ManTra-Net trainning on NIST16 {} dataset'.format(parManager.EpochDone, dataSetChoosen))
    plt.text(0.7, 0.3,r'$AUC$:{:.6F}'.format(AUC))
    
    plt.plot(FPR, TPR) # front parameter is for x, back parameter is for y
    plt.show()
    

    