# Remote_sensing-forrests. 
## Unsupervised ML method is applied to create a classificator of forrests on LANDSAT images. Invariant Information Clustering algorithm is in use.
### DATA:
Directory DATA/data_raw should contain a number of LANDSAT scenes as .tar archives. Example of the file is stored in the dir with only 1 band (channel).

### RawDataProcessing:
contains a code for making HDF5 file from raw images from DATA containig set of images in 7 spectral channels (datasets of shape (N,SIZE,SIZE,7)).
Eventual HDF5 file is stored at './DATA/h5_files'.

### TrainingNN
contains scripts for creating and training a neural network.

### main.py:
creates and trains a ResNet neural network using TrainingNN scripts.


### Some .py scripts have .ipynb clones.
