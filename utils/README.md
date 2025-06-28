# utils README

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
-	`math`
-	`matplotlib.pyplot`
-	`scipy.signal` (specifically `butter`, `lfilter`, `buttord`, `freqz`)
-	`scipy.stats` (specifically `norm`)
-	`random`

## plot_feat_32x40.py

### Description

This Python script plots a 32x40 feature vector for a specified bird species, source file, and audio segment. It locates the most recent feature data file (feats_32x40*.csv) and its corresponding identifier file (bvp_ids*.csv), finds the specific feature vector based on the command-line inputs, scales the vector, and then generates an image plot using matplotlib.
### Command Line Usage

The script requires three arguments to be passed at the command line.

python plot_feat_32x40.py <bird_species> <filename> <segment_num>

Inputs

    <bird_species>

        Description: The target bird species. This corresponds to a directory name within the ../dataset/audio/ path where the analysis files are stored.

        Type: string

    <filename>

        Description: The name of the original source audio file (e.g., XC12345) that the desired segment belongs to.

        Type: string

    <segment_num>

        Description: The specific segment number from the audio file to be plotted.

        Type: integer

### Outputs
Primary Output

    Feature Vector Plot: The script generates and displays a plot of the requested 32x40 feature vector. The script will pause and wait for user input in the terminal before closing the plot and terminating.

Console Output

The script prints the following information to the console during its execution:

    Confirmation of Inputs: Displays the arguments passed to the script.

    bird_species: [species_name] src_file: [file_name] segment_num: [segment_number]

    User Prompt: After displaying the plot, the script prompts the user to continue.
	
##scatterplot.py

### Description

This Python script displays a scatter plot of a 2D encoding saved in a column format with headings named "encoded_" followed by a number between 0 and z_dim-1. If the latent dimension (`z_dim`) is greater than 2, t-SNE (t-distributed Stochastic Neighbor Embedding) is applied to reduce the dimensionality to 2 for visualization.

The script reads a `.csv` file containing the encodings, generates a scatter plot, and color-codes the data points based on a specified column. The plot is then displayed in a window.

### Command Line Inputs

| Argument | Abbreviation | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `species` | | string | | The bird species to process. This is used to locate the data file within the directory structure. |
| `filename` | | string | | The name of the `.csv` file that contains the encodings. |
| `--color-col`| `-c` | string | `meta_type` | The name of the column in the `.csv` file to use for color-coding the scatter plot points. |
| `--point-sz` | `-s` | integer | `14` | The size of the markers in the scatter plot. |
| `--z-dim` | `-z` | integer | `10` | The input dimensionality of the encodings. |

### Outputs

* **Console Output:**
    * Prints the path of the file being processed.
    * If the file cannot be read, it prints an error message and exits.
    * If `z_dim` > 2, it displays the progress of the t-SNE algorithm.

* **Graphical Output:**
    * A window displaying a 2D scatter plot of the encodings.
    * The plot includes a legend, with colors corresponding to the unique values in the column specified by `--color-col`.
    * The plot has a grid with yellow lines.	
	
	
# sample_outliers.py

### Description

This Python script facilitates the analysis of audio data by randomly sampling detected outliers from a specified `discards_*.csv` file. It presents each outlier to the user, who can then confirm or reject the outlier designation, or mark it as indeterminate. The script is designed to achieve a 95% confidence level with a 5% margin of error in its sampling.

For each randomly selected sample that hasn't been previously evaluated, the script performs the following actions:
1.  Plots the audio segment from which the sample originates, highlighting the specific sample section.
2.  Plays the audio of the entire segment.
3.  Prompts the user to classify the sample as a true positive, false positive, or undetermined, and allows for an optional comment.
4.  Records the user's classification and any accompanying comments in new 'outlier' and 'comment' columns within the `discards` table.
5.  This process is repeated until the target number of samples (plus any undetermined decisions) has been reviewed.
6.  Finally, it calculates the precision of the outlier detection and outputs the updated `discards` table along with the computed precision.

### Command Line Inputs

-   `species`: (string) The bird species for which to sample outliers.
-   `filename`: (string) The name of the `.csv` file that contains the outlier data.
-   `--outlier-cols` or `-o`: (string, optional) A comma-separated list of column names that contain the outlier designations. The default value is 'vade_maj'.
-   `--proportion` or `-p`: (float, optional) The assumed outlier proportion used for calculating the necessary sample size. The default is 0.5.
-   `--out`: (string, optional) The name of the output file for the summary. The default is 'sample_outliers.out'.
-   `--scope` or `-s`: (string, optional) Defines the scope of the sampling. It can be 'stat' for statistical sampling, 'all' to sample all flagged outliers, or 'test' to run a test of the post-sampling backend code. The default is 'stat'.

### Outputs

-   **`<species>/analysis/artifacts/<filename>`**: The original input `discards_*.csv` file is updated with two new columns:
    -   `outlier`: Stores the user's classification of the sample (1 for outlier, 0 for inlier, -1 for indeterminate).
    -   `comment`: Contains any optional text comments provided by the user.
-   **`<species>/analysis/artifacts/<out_file>`**: A summary file (default name `sample_outliers.out`) is generated, containing the following information:
    -   A header section detailing the input parameters used for the script execution.
    -   For each outlier column analyzed:
        -   The calculated precision of the outlier detection.
        -   The number of samples that were evaluated.
        -   The number of true positives identified within the sample.
        -   The margin of error at a 95% confidence level.
        -   An estimated range of the total number of true positives in the entire outlier population.
		
## sample_inliers.py

### Description

This Python script facilitates the manual review of audio segments that have been automatically classified as "inliers" (i.e., not outliers). It randomly selects a statistically significant number of these inlier samples from a given CSV file. For each selected sample, the script displays a spectrogram of the audio segment and plays the corresponding sound. The user is then prompted to confirm or reject the inlier classification, or mark it as undetermined.

The primary goal of this process is to calculate the false negative rate of the initial classification. The script adds the user's evaluation and any comments to the input CSV file. Finally, it generates a summary file containing the calculated false negative rate, margin of error, and an estimated range of total false negatives within the entire dataset.

### Command Line Inputs

-   `species`: (Positional, string) The bird species for which the inliers are being sampled.
-   `filename`: (Positional, string) The name of the .csv file located in the `../dataset/audio/<species>/analysis/artifacts/` directory that contains the inlier data to be sampled.
-   `--inlier-cols` or `-o`: (Optional, string, default: 'vade_maj') A comma-separated list of column names in the input CSV that contain the inlier designations.
-   `--proportion` or `-p`: (Optional, float, default: 0.1) The assumed inlier proportion used for calculating the required sample size.
-   `--out`: (Optional, string, default: 'sample_inliers.out') The name of the output file where the summary statistics will be saved.
-   `--scope` or `-s`: (Optional, string, default: 'stat') Defines the scope of the sampling. It can be one of three options:
    -   `stat`: Performs a statistical sampling to achieve a 95% confidence level with a 5% margin of error.
    -   `all`: Processes all inliers for the specified columns.
    -   `test`: Skips the interactive user evaluation and runs the backend calculations, which is useful for testing.

### Outputs

1.  **Updated Discards CSV File**: The original input CSV file (specified by `filename`) is updated with the following columns:
    * `inlier`: An integer representing the user's classification: `1` for a true inlier (correctly classified), `0` for an outlier (a false negative), and `-1` for an undetermined case.
    * `comment`: An optional text comment provided by the user for each sample.

2.  **Summary Output File**: A text file (with the name specified by the `--out` argument) is created in the `../dataset/audio/<species>/analysis/artifacts/` directory. This file contains a summary of the sampling process and its results, including:
    * A header section detailing the input files and parameters used for the script execution.
    * For each inlier column analyzed:
        * The assumed proportion and the calculated sample size.
        * The False Negative Rate (FNR) calculated from the user's classifications.
        * The number of samples that were actually evaluated.
        * The number of false negatives found within the sample.
        * The margin of error for the calculated FNR (at a 95% confidence level).
        * An estimated range of the total number of false negatives in the entire population of inliers for that column.
		
## shannon_ent.py

### Description

This script computes descriptive statistics over the per-clip Shannon entropy for a specified bird species. It processes feature files, calculates the entropy for each clip, and then provides overall statistics as well as a breakdown of statistics for different vocalization types (call, song, other).

### Command Line Inputs

-   `species`: The bird species for which to perform the processing. This is a required positional argument.

### Outputs

The script prints the following information to the console:

-   The path of the features file being processed.
-   The path of the BVP (Bird Vocalization Phrase) IDs file being processed and its dimensions.
-   If the `meta_type` column is not in the BVP IDs file, it will print the name of the BVP file from which it is retrieving this information.
-   Overall descriptive statistics for the Shannon entropy of all clips, including:
    -   `count`: The total number of clips.
    -   `mean`: The mean Shannon entropy.
    -   `std`: The standard deviation of the Shannon entropy.
    -   `min_val`: The minimum Shannon entropy.
    -   `Q1`: The first quartile of the Shannon entropy.
    -   `median`: The median Shannon entropy.
    -   `Q3`: The third quartile of the Shannon entropy.
    -   `max_val`: The maximum Shannon entropy.
-   If there are clips labeled as 'call', it will print the same set of descriptive statistics for just those clips under the heading `*** CALL label entropy stats ***`.
-   If there are clips labeled as 'song', it will print the same set of descriptive statistics for just those clips under the heading `*** SONG label entropy stats ***`.
-   If there are clips labeled as 'other', it will print the same set of descriptive statistics for just those clips under the heading `*** OTHER label entropy stats ***`.
-   Error messages will be printed if necessary files (features or BVP files) are not found or if there is an error opening them.
-   A message is printed if a `NaN` (Not a Number) value is encountered during entropy calculation, indicating the species and the index of the problematic data.		

## shannon_ent_bulk.py

### Description

This script computes descriptive statistics over the per clip Shannon entropy for each of a input list of bird species and outputs results for individual species in the `<bird_species>\analysis\artifacts` folder and collective results in shannon_ent.csv. This script is based on shannon_ent.py which only computes the metric for a single species.

The script calculates the Shannon entropy for each audio clip feature, which is a measure of the uncertainty or randomness in the signal. It then computes and saves descriptive statistics (count, mean, standard deviation, min, quartiles, median, max) for these entropy values, both for all clips of a species and separately for clips labeled as 'call', 'song', or 'other'.

### Command Line Inputs

* `list_file`: (string)
    * text file containing list of bird species to perform processing for.

### Outputs

* **`<bird_species>/analysis/artifacts/shannon_ent.out`**:
    * A text file generated for each bird species processed.
    * Contains the descriptive statistics of the Shannon entropy for all audio clips of that species.
    * Also contains descriptive statistics for subsets of clips labeled as 'call', 'song', and 'other'.
    * The statistics include: count, mean, std, min_val, Q1, median, Q3, and max_val.

* **`shannon_ent_bulk.out`**:
    * A CSV file that collects the overall mean and median Shannon entropy for each processed bird species.
    * Each line in the file corresponds to a species and has the format: `species_name, mean_entropy, median_entropy`.