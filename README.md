# Remote_sensing-forrests
Unsupervised ML method is applied to create a classificator of forrests on LANDSAT images.

Directory "MyScripts" contains of usefull scripts. "main.py" creates and trains a ResNet neural network. Directory "MyNotebooks" contains some jupyter notebooks, which are basically drafts for scripts. Training data must be represented with .h5 file, containig set of 512x512 images in 7 spectral channels (datasets of shape (N,512,512,7) ). The HDF5 file must be located by path "../DATA/h5_files/".
