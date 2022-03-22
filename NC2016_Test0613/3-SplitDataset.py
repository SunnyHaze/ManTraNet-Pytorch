import csv
from random import shuffle
DIR = '../NIST2016/'
IndexDir = DIR + 'index.csv'
TrainDir = DIR + 'Train.csv'
TestDir = DIR + 'Test.csv'
SplitRate = 0.8   # rate between trainning set and the whole dataset

# Read data
data = []
with open(IndexDir, 'r') as f:
    reader = csv.reader(f)
    for i in reader:
        data.append(i)

def countN(array):
    cnt = 0
    for i in array:
        if i[1] == 'N':
            cnt += 1
    return cnt

cnt = countN(data)
length = len(data)
print('---Raw Dataset---')
print("There are {} pictures in total.".format(length))
print("There are {:.2f}% of No-manipulated pictures while {:.2f}% of manipulated.".format(cnt/ length * 100, 100 - cnt/length * 100))

print('---Splited dataset---')
shuffle(data)
splitPoint = int(length * SplitRate) + 1
print("len(Train.csv): {}".format(splitPoint))
print("len(test.csv): {}".format(length - splitPoint))

Train = data[:splitPoint]
Test = data[splitPoint:]
print('---Trainning set proportion---')
cnt = countN(Train)
length = len(Train)
print("There are {:.2f}% of No-manipulated pictures while {:.2f}% of manipulated.".format(cnt/ length * 100, 100 - cnt/length * 100))
print('---Testing set proportion---')
cnt = countN(Test)
length = len(Test)
print("There are {:.2f}% of No-manipulated pictures while {:.2f}% of manipulated.".format(cnt/ length * 100, 100 - cnt/length * 100))


def saveCSV(filePath, array):
    with open(filePath, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(array)
        
saveCSV(TrainDir, Train)
saveCSV(TestDir, Test)
