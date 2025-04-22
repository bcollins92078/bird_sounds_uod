# bird_sounds_uod (repository under construction)
This repository contains python code developed to preprocess bird sounds audio files downloaded from [Xeno-Canto](https://xeno-canto.org/) and perform deminsionality reduction and unsupervised outlier detection (UOD) on single species datasets. Three model types are implemented.
* Convolutional Variational Autoencoder (CVAE) followed by Hierarchical Agglomerative Clustering (HAC)
* Convolutional Autoencoder (CAE) followed by HAC
* Variational Deep Embedding (VaDE) 

## Dependencies
1. Install [Tensorflow](https://www.tensorflow.org/get_started/os_setup)
2. Install [Matplotlib](https://matplotlib.org/index.html)
3. Install [Numpy](http://www.numpy.org/)
4. Install [Pandas](https://pandas.pydata.org/)
