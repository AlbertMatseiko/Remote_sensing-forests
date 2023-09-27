# Remote_sensing-forrests. 
## Unsupervised ML method is applied to create a classificator of forrests on LANDSAT images. Invariant Information Clustering algorithm is in use.
### DATA:
Directory DATA/data_raw should contain a number of LANDSAT scenes as .tar archives. Example of the file is stored in the dir with only 1 band (channel).
More raw data samples one can find by link https://drive.google.com/drive/folders/1NO5xrnibQDGEkF38lepmGcsank3Pqr-x?usp=sharing.

### RawDataProcessing:
contains a code for making HDF5 file from images from DATA. Eventual .h5 file contains set of images in 7 spectral channels (datasets of shape (N,SIZE,SIZE,7)) and is stored at './DATA/h5_files'.

### TrainingNN
contains scripts for creating and training a neural network.

### main.py:
creates and trains a ResNet neural network using TrainingNN scripts.


### Some .py scripts have .ipynb clones.
