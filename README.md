# UOD_bird_sounds (repository under construction)
This repository contains python code developed in conjunction with the project reported on in http://arxiv.org/abs/2504.18650. This code preprocess bird sounds audio files downloaded from [Xeno-Canto](https://xeno-canto.org/) and perform deminsionality reduction and unsupervised outlier detection (UOD) on single species datasets. Three model types are implemented.
* Convolutional Variational Autoencoder (CVAE) followed by Hierarchical Agglomerative Clustering (HAC)
* Convolutional Autoencoder (CAE) followed by HAC
* Variational Deep Embedding (VaDE) 
In addition a number of utilities are included in this repository.

## segment_audio.py  
Python script segments each audio recording for a specified bird species.

Inputs:
* bird species (common name used for folder names)

Outputs:
* <bird_species>_bvp.csv file containing metadata for the segmented audio, one segment per row, with the following columns
    + src_file	
    + segment_num	
    + num_segments	
    + time_offset	
    + duration	
    + sinr	

## extract_feats.py
This file contains a Python script which is executed per species to extract fixed length features (mel spectrograms) of specified resolution that are used for downstream processing (e.g., dimensionality reduction and clustering).

## cvae_clean.py 
This python script trains CVAE models on the one species dataset specified in calling argument.
- Goal is to identify outliers by 
     * dimensionality reduction by training set of CVAE models on species dataset 
     * applying hierarchical agglomerative clustering (HAC) on latent space of each CVAE model
     * flagging the most distant clusters from the nearest "big cluster" and then smallest first for discard until max discards are achieved
     * summing per sample discard recommendations across models and applying majority vote criterion to arrive at final discard recommendations

- Parameters for trained models are saved in the <species>\analysis\models folder

- various artifacts are saved in <species>\analysis\artifacts folder
    * cvae model encodings files (encodings_clustered_cvae_a2_z<z_dim>_<timestamp>.csv)
    * summary of models trained during run of cvae_clean.py (summary_cvae_a2_z<z_dim>_<timestamp>.csv)
## vade_clean.py 
This python script trains vade models on the one species dataset specified in calling argument. It also trains CAE "pretraining" models in the process. 
- The goal is to identify outliers by majority vote across the vade models and, since pretraining models are a byproduct, apply HAC to identify small, late merging clusters and majority vote across these as well. 
- Parameters for trained models are saved in the <species>\analysis\models folder 
     * pretraining models (pretrained_model_<timestamp>.pth)
     * trained vade models (trained_model_<timestamp>.pth)

- various artifacts are saved in <species>\analysis\artifacts folder
     * vade model encodings files (vade_ec_c<n_clusters>_z<z_dim>_<timestamp>.csv)
     * summary of models trained during run of vade_clean.py (summary_c<n_clusters>_z<z_dim>_<timestamp>.csv)
