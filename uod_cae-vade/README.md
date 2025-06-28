# uod_cae-vade README
This directory contains python scripts that perform the CAE and VaDE base UOD reported on in http://arxiv.org/abs/2504.18650.

## Dependencies
These scripts require the following Python libraries:

-   `os`
-   `sys`
-   `glob`
-   `pathlib`
-   `pandas`
-   `numpy`
-   `argparse`
- 	`time`
-	`datetime`
-	`warnings`
-	`librosa`
-	`torch`
-	`torchvision`
-	`scipy.optimize` (specifically `opt`)
-	`scipy.cluster.hierarchy` (specifically `linkage`, `cophenet`, `fcluster`)
- 	`scipy.spatial.hierarchy` (specifically `pdist`, `squareform`)
-	`sklearn.mixture` (specifically `GaussianMixture`)
-	`sklearn.metrics.cluster` (specifically `homogeneity_score`)
-	`array`

## vade_clean.py

### Description

This Python script performs data UOD (i.e., cleaning) for the spectrogram dataset for a single species using a convolutional autoencoder (CAE) and Variational Deep Embedding (VaDE).

**UOD using CAE:**
* **Dimensionality Reduction:** A CAE is trained on preprocessed spectrogram clips from bird sound recordings. The parameters from this model are also used as a starting point for VaDE model training (i.e., pre-training).
* **Clustering:** Hierarchical Agglomerative Clustering (HAC) is applied to the CAE's latent space representation of the clips to produce `n_clusters`.
* **Discard Recommendations:** Candidate clips for discarding are identified at the cluster level. The smallest clusters farthest away from one of the largest clusters are recommended for discard first. This process continues with larger clusters until a specified maximum discard percentage is reached.
* **Iteration:** The preceding steps are repeated a specified number of times. Each iteration saves the model parameters and candidate discards.

**UOD using VaDE:**
* **Model Training:** Train VaDE model on a one-species preprocessed spectrograms clip dataset using parameters from a CAE model trained on the same dataset as a starting point
* **Outlier Detection:** For each clip in the dataset calculate the highest probability of membership across the GMM clusters identified. Clips are designated as candidate outliers starting with the lowest probability of cluster membership until the total number of candidates reaches the specified maximum number of discards.

### Command Line Inputs

* `species`
    * **Description:** The bird species to process.
    * **Type:** string
* `--pretrain-epochs` or `-p`
    * **Description:** Number of epochs for CAE pre-training.
    * **Type:** int
    * **Default:** 20
* `--epochs` or `-e`
    * **Description:** Number of epochs for VaDE training.
    * **Type:** int
    * **Default:** 20
* `--gpu` or `-g`
    * **Description:** The ID of the GPU to use. A negative number indicates the CPU will be used.
    * **Type:** int
    * **Default:** -1
* `--learning-rate` or `-l`
    * **Description:** The learning rate for the optimizer.
    * **Type:** float
    * **Default:** 0.001
* `--batch-size` or `-b`
    * **Description:** The number of samples per batch.
    * **Type:** int
    * **Default:** 128
* `--n-clusters` or `-c`
    * **Description:** The number of clusters (classes) for the models.
    * **Type:** int
    * **Default:** 2
* `--n-types` or `-t`
    * **Description:** The number of label values (e.g., song, call, other).
    * **Type:** int
    * **Default:** 4
* `--z-dim` or `-z`
    * **Description:** The dimension of the latent variable space.
    * **Type:** int
    * **Default:** 10
* `--iterations` or `-i`
    * **Description:** The number of times to train and evaluate the models.
    * **Type:** int
    * **Default:** 11
* `--song-call` or `-s`
    * **Description:** Specifies if the species has distinct song and call vocalizations (1 for yes, 0 for no). This affects the calculation of summary metrics.
    * **Type:** int
    * **Default:** 1
* `--max-discards` or `-d`
    * **Description:** The maximum fraction of samples to be recommended for discarding.
    * **Type:** float
    * **Default:** 0.1

### Outputs

The script generates several files for each run, which are saved in the `../dataset/audio/<species>/analysis/` directory. The filenames include a timestamp (`YYYYMMDD-HHMMSS`) from the last successful iteration.

* **Pretrained CAE Model**
    * **Path:** `models/pretrained_model_c<n_clusters>_z<z_dim>_<timestamp>.pth`
    * **Description:** The saved state dictionary of the trained Convolutional Autoencoder (CAE) for each iteration. This model is used to initialize the VaDE model.
* **Trained VaDE Model**
    * **Path:** `models/vade_model_c<n_clusters>_z<z_dim>_<timestamp>.pth`
    * **Description:** The saved state dictionary of the trained VaDE model for each successful iteration.
* **VaDE Clustered Encodings**
    * **Path:** `artifacts/vade_ec_c<n_clusters>_z<z_dim>_<timestamp>.csv`
    * **Description:** A CSV file generated for each iteration, containing detailed output for every input clip. Columns include source file, segment number, latent space encodings, probabilities for each cluster, the final assigned cluster, and the maximum cluster probability.
* **Anomaly Scores**
    * **Path:** `artifacts/anomal_c<n_clusters>_z<z_dim>_<timestamp>.csv`
    * **Description:** A consolidated CSV file that stores the outlier scores for each clip across all iterations. It includes binary discard recommendations from the CAE+HAC pretraining step and the maximum cluster membership probability from the VaDE model for each iteration.
* **Summary Statistics**
    * **Path:** `artifacts/summary_c<n_clusters>_z<z_dim>_<timestamp>.csv`
    * **Description:** A CSV file that summarizes the performance metrics for each iteration. This includes training/validation loss, cluster size distribution, song/call purity for each cluster (if applicable), and the homogeneity score.
	
## vade_post.py

### Description

This script post-processes summary and anomal files from vade_clean to arrive at aggregated discard recommendations as follows:
* Model selection criteria are applied to training results from both VADE and VAE pretraining models to determine which models to use anomalousness scores from.
* Aggregate anomalousness scores from VADE and CVAE pretraining models that pass selection criteria and make discard recommendations.

### Command-Line Inputs

* `species`
    * **Description:** Bird species to post-process VADE outputs for.
    * **Type:** string
* `--hi-size-thresh`, `-s`
    * **Description:** Model largest cluster size fraction above median selection criteria (-1 disables).
    * **Type:** float
    * **Default:** 0.2
* `--hi-loss-thresh`, `-l`
    * **Description:** Training/test loss fraction above median selection criteria (-1 disables).
    * **Type:** float
    * **Default:** 0.2
* `--n-clusters`, `-c`
    * **Description:** Number of classes.
    * **Type:** int
    * **Default:** 2
* `--z-dim`, `-z`
    * **Description:** Number of latent dimensions.
    * **Type:** int
    * **Default:** 10
* `--max-discard-frac`, `-d`
    * **Description:** Max fraction of samples to discard.
    * **Type:** float
    * **Default:** 0.1
* `--degen-pct-size-thresh`, `-b`
    * **Description:** Degenerate model largest cluster absolute percentage size threshold to exclude model.
    * **Type:** float
    * **Default:** 0.85

### Outputs

#### File Output

This script does not create new files but modifies the most recent `discard_fix_c(n_cluster)_z(z_dim>_*.csv` file located in the `<species>/analysis/artifacts/` directory. The script adds the following columns to this file:

-   `pre_tally`: The total count of discard recommendations a clip received from all selected CAE model iterations.
-   `pre_maj`: The final CAE discard recommendation (True/False) based on a majority vote threshold.
-	`vade_tally`: The total count of discard recommendations a clip received from all selected VaDE model iterations.
-   `vade_maj`: The final VaDE discard recommendation (True/False) based on a majority vote threshold.

#### Console Output

* A summary of the provided command-line arguments.
* The number and labels of pretraining models excluded based on cluster size criteria.
* The total number of VADE models excluded, with a breakdown and list of labels for models excluded due to high training loss, high test loss, or large cluster size.
* The number of selected pretrain and VADE models used for aggregation.
* Statistics on the final discard recommendations:
    * The total number of samples and the target maximum number of discards.
    * The number and percentage of samples recommended for discard by a majority of pretraining models.
    * The number and percentage of samples recommended for discard by a majority of VADE models.
    * The number and percentage of samples recommended for discard by either pretrain OR VADE majority vote.
    * The number and percentage of samples recommended for discard by both pretrain AND VADE majority vote.
    * The correlation between the pretrain and VADE majority discard recommendations.
* If applicable, the average `homogeneity_score` across all selected VADE models.

