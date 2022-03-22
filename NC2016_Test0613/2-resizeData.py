import string
import pandas
from PIL import Image
import numpy as np
import csv
paths = {'mani' : './reference/manipulation/NC2016-manipulation-ref.csv',
  'removal' : './reference/remove/NC2016-removal-ref.csv',
  'splice' : './reference/splice/NC2016-splice-ref.csv'
  }
globalIdx = 0
SavePath = '../NIST2016_500/'
TargetSize = 512

TableOfContent = []
selected = []
# mani removal splice
def decodeByTask(taskID : string, idx):
  print('--Start task {}---'.format(taskID))
  table = pandas.read_csv(paths[taskID])
  
  # =====pictures with masks=====
  manupilated = table[table['IsTarget']== 'Y']
  manupilated = manupilated[['ProbeFileName','ProbeMaskFileName']]
  length = len(manupilated)
  
  for i in range(length):
    if i % 50 == 0:
      print('{:.4f}% Has done'.format(i/len(manupilated) * 100))
      
    image_dir = manupilated.iloc[i]['ProbeFileName']
    if image_dir in selected:
      continue
    else:
      mask_dir = manupilated.iloc[i]['ProbeMaskFileName']
      selected.append(image_dir)
      
      image = Image.open(image_dir)
      (x, y) = image.size
      y = int(y/x*512)
      image = image.resize((TargetSize, y), Image.ANTIALIAS)
      image_name = '{}.jpg'.format(idx)
      image.save('{}{}.jpg'.format(SavePath, idx))
      
      mask = Image.open(mask_dir)
      (x, y) = image.size
      y = int(y/x*512)
      mask = mask.resize((TargetSize,y),Image.ANTIALIAS)
      mask_name = '{}_mask.jpg'.format(idx)
      mask.save('{}{}_mask.jpg'.format(SavePath, idx))
      
      
      TableOfContent.append([image_name, mask_name, taskID])
      idx +=1
  
  # ======pictures without masks======
  manupilated = table[table['IsTarget']== 'N']
  manupilated = manupilated[['ProbeFileName']]
  length = len(manupilated)

  for i in range(length):
    if i % 50 == 0:
      print('{:.4f}% Has done'.format(i/len(manupilated) * 100))
      
    image_dir = manupilated.iloc[i]['ProbeFileName']
    if image_dir in selected:
      continue
    else:  
      image = Image.open(image_dir)
      (x, y) = image.size
      y = int(y/x*512)
      image = image.resize((TargetSize, y),Image.ANTIALIAS)
      image_name = '{}.jpg'.format(idx)
      image.save('{}{}.jpg'.format(SavePath, idx))
      
      TableOfContent.append([image_name, 'N', taskID])
      selected.append(image_dir)
      idx +=1
  return idx

def decodeNist():
  globalIdx = 0
  globalIdx = decodeByTask("mani", globalIdx)
  globalIdx = decodeByTask("removal", globalIdx)
  globalIdx = decodeByTask("splice", globalIdx)
  with open("{}index.csv".format(SavePath), 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(TableOfContent)
    
if __name__ == '__main__':
  decodeNist()