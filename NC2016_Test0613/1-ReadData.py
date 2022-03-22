import pandas,random
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
paths = {'mani' : './reference/manipulation/NC2016-manipulation-ref.csv',
  'removal' : './reference/removal/NC2016-removal-ref.csv',
  'splice' : './reference/splice/NC2016-splice-ref.csv'
  }
table = pandas.read_csv(paths['splice'])
print(table)
manupilated = table[table['IsTarget']== 'Y']
manupilated = manupilated[['ProbeFileName','ProbeMaskFileName']]
print(manupilated)
print(manupilated[563:564])
print(len(manupilated))

idx = random.randint(0,len(manupilated) - 1 - 10)
for i in range(5):
  data = manupilated.iloc[idx + i]
  print(data)
  image_dir = data['ProbeFileName']
  mask_dir = data['ProbeMaskFileName']
  image = Image.open(image_dir)
  image = np.array(image)
  plt.subplot(2,5,i + 1)
  plt.imshow(image)
  
  mask = Image.open(mask_dir)
  mask = np.array(mask)
  plt.subplot(2,5 ,i + 5 + 1)
  plt.imshow(mask,cmap='gray')

plt.show()