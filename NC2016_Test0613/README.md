# README for NIST16 DATASET
This Dir is used to place the dataset from NIST16, To reduce the size of git repo, I added all the *jpg files to the ".gitigore".

If you want to perform Mantra-Net with NIST16 data, you can download it on the official website of [NIST](https://www.nist.gov/itl/iad/mig/nimble-challenge-2017-evaluation), and view the data file or resize the file to (256, 256) with Python scriprt given in this dir:
- [ReadData.py](ReadData.py) Shows and illustrate part of data in NIST16 dataset
- [resizeData.py](resizeData.py) Will resize the raw pictures to a smaller scale in order to accelerate the training. However, in our test, resize can eliminate a large number of features used to identify modifications, which means reduce the performance of the net. You could set parameters and scale you like to reproduce your own sub-dataset. 