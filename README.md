# UOD_bird_sounds (repository under construction)
This repository contains python code developed in conjunction with the project reported on in http://arxiv.org/abs/2504.18650. This code preprocesses bird sounds audio files downloaded from [Xeno-Canto](https://xeno-canto.org/) and perform dimensionality reduction and unsupervised outlier detection (UOD) on single species datasets. Three model types are implemented.
* Convolutional Variational Autoencoder (CVAE) followed by Hierarchical Agglomerative Clustering (HAC)
* Convolutional Autoencoder (CAE) followed by HAC
* Variational Deep Embedding (VaDE) 

This work assumes that bird audio recordings have been downloaded from Xeno-Canto using the API wrapper documented at https://pypi.org/project/xeno-canto/

The repository is organized in the following subfolders
* /preprocessing subfolder contains scripts that perform the audio preprocessing 
* /uod_cvae subfolder contains scripts that perform UOD on preprocessed audio by training CVAE models, applying HAC to the latent space of each model and then aggregating the outlier designations through majority voting.
* /uod_cae-vade subfolder contains scripts that perform UOD on preprocessed audio by training VaDE models, designating outlier candidates based on minimum distance to a cluster and then aggregating the candidate designations through majority voting. Since a CAE model is also trained as pre-training for each VaDE model, HAC is also applied to the latent space of each of these CAE models and aggregated outlier designations are generated in a similar fashion as done for CVAE.
* /utils subfolder contains some utility scripts that were found to be useful 

Each of these subfolders contain a README.md file that provide some additional documentation specific the scripts contained there.

All of the above scripts expect the data to be stored in a folder hierarchy as follows:
* ../dataset
* ../dataset/metadata
* ../dataset/metadata/<bird_species> - Xeno-Canto recording file metadata is downloaded; <bird_species> is the English common name for a bird species for which audio recordings are downloaded
* ../dataset/audio
* ../dataset/audio/<bird_species> - where <bird_species> is the English common name for a bird species for which audio recordings are downloaded. Xeno-Canto utilities will create these folders as part of downloading.
* ../dataset/audio/<bird_species>/analysis - some of the preprocessing artifacts are stored here
* ../dataset/audio/<bird_species>/analysis/artifacts - UOD artifacts are generally stored here
* ../dataset/audio/<bird_species>/analysis/models - model training artifacts are stored here

