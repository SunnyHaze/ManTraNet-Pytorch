# README for NIST16 DATASET
This Dir is used to place the dataset from NIST16, To reduce the size of git repo, I added all the *jpg files to the ".gitigore".

If you want to perform Mantra-Net with NIST16 data, you can download it on the official website of [NIST](https://www.nist.gov/itl/iad/mig/nimble-challenge-2017-evaluation), and view the data file or resize the file to (256, 256) with Python scriprt given in this dir:
- [1-ReadData.py](1-ReadData.py) Shows and illustrate part of data in NIST16 dataset
- [2-resizeData.py](2-resizeData.py) Will resize the raw pictures to a smaller scale in order to accelerate the training. However, in our test, resize can eliminate a large number of features used to identify modifications, which means reduce the performance of the net. You could set parameters and scale you like to reproduce your own sub-dataset. This script will also create an `index.csv` file to make an index for the output images in directory `./NIST2016`.
- [3-SplitDataset.py](3-SplitDataset.py) This Python file will create two split index files **randomly** for Training dataset and Testing dataset, respectively. It is generated from `./NIST2016/index.csv` file, so you must run `2-resizeData.py` in advance to create `Train.csv` and `Test.csv` correctly.
  > The `Train.py` in root directory needs to read `Train.csv` and `Test.csv` to run the trainning process. 