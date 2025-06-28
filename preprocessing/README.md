# Preprocessing README
This directory contains python scripts that perform the preprocessing steps reported on in http://arxiv.org/abs/2504.18650.

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
-	`itertools`
-	`scipy.signal` (specifically `butter`, `lfilter`, `buttord`, `freqz`)


## segment_audio.py

### Description

This Python script automates the extraction of Bird Vocalization Phrases (BVPs) from each audio recording for a specified bird species. It processes all `.mp3` files in a species-specific directory, segments them to remove silence based on short-term energy, and calculates the Signal-to-Interference-plus-Noise Ratio (SINR) for each segment. The script flags segments that may require manual inspection based on their length or frequency of occurrence.

### Command Line Inputs

The script accepts the following command-line arguments:

* `species`
  * **Description**: The bird species to process BVPs for. This is a required positional argument and should match the folder name containing the audio files.
  * **Type**: string
* `--min-silence-len`
  * **Description**: The minimum duration of silence, in time frames, required to separate two distinct vocalization segments.
  * **Type**: int
  * **Default**: 11
* `--min-freq`
  * **Description**: The minimum frequency (in Hz) to be considered in the analysis. A high-pass filter is designed based on this value to remove low-frequency noise.
  * **Type**: int
  * **Default**: 1000

#### Example Usage
```bash
python segment_audio.py HouseFinch --min-silence-len 15 --min-freq 1200
```
### Outputs

The script produces two types of output: a CSV data file containing the segmentation results and messages printed to the console.

#### BVP Data File
A single CSV file is generated in the `../dataset/audio/<species>/analysis/` directory.

* **Filename**: `<bird_species>_bvp_<YYYYMMDD>.csv`
* **Description**: This file contains the details of each segmented Bird Vocalization Phrase (BVP) found in the audio files. Each row represents one BVP.
* **Columns**:
  * `src_file`: The name of the source audio file from which the segment was extracted.
  * `segment_num`: A sequential number for each BVP within a single source file.
  * `num_segments`: The total number of BVPs extracted from the source file.
  * `time_offset`: The start time (in seconds) of the BVP within the source audio file.
  * `duration`: The duration (in seconds) of the BVP.
  * `sinr`: The Signal-to-Interference-plus-Noise Ratio (in dB) of the BVP, calculated relative to the average energy of the non-BVP parts of the file.
  * `inspect`: A flag to suggest manual inspection. It can contain values like 'one segment', 'many segments', or 'long segment' if the segmentation results are unusual.
  * `meta_type`: The recording content type, populated from an external `local_birds_meta.csv` file (e.g., 'song', 'call').

#### Console Output
The script prints the following information to the standard output during execution:

* **Run Arguments**: A summary of the arguments the script was run with is printed at the beginning.
* **Status Messages**: For certain files, the script will print a notice that the file has been flagged for inspection (e.g., `... flagged ... one segment`).
* **Error Messages**: File read errors or other exceptions are printed to the console.
* **Final Summary**: Upon completion, the script prints the total number of BVPs found and their total cumulative duration.
* **Execution Time**: The total time taken for the script to run is printed at the very end.

## add_metadata.py

### Description

This script adds metadata from `local_birds_meta.csv` to the specified species' `_bvp.csv` file. It reads the most recent `_bvp.csv` file for a given bird species, looks up corresponding metadata for each audio file listed, adds this metadata as a new column, and then saves the result to a new timestamped CSV file.

### Command Line Inputs

* `bird species common name`: The common name of the bird species to process (e.g., "AmericanRobin").

### Outputs

* `updated <bird_species>_bvp_<timestamp>.csv file`: A new CSV file is generated in the `<species_name>/analysis/` directory. This file contains all the original data from the input `_bvp.csv` file, plus an additional `meta_type` column with the added metadata. The timestamp ensures that previous files are not overwritten.


## extract_feats.py

### Description
This file contains a Python script which is executed per species to extract fixed length features of specified resolution that are used for downstream processing (e.g., dimensionality reduction and clustering). It is evolved from cluster_feat.py and hi_res_feat.py.

### Command Line Inputs
`species`
- description: bird species to process BVPs for
- type: str

`--sinr`
- description: minimum SINR (dB)
- type: float
- default: 10

`--fg-thresh`
- description: foreground threshold
- type: float
- default: 5

`--n-frms` or `-f`
- description: number of time frames
- type: int
- default: 40

`--n-mels` or `-m`
- description: number of mels (freq bins)
- type: int
- default: 32

`--min-freq`
- description: minimum frequency
- type: int
- default: 1000

#### Example Usage
```bash
python extract_feats.py HouseFinch --sinr 12 --fg-thresh 3 -f 40 -m 32 --min-freq 1500
```

### Outputs
1.  **BVP IDs File**
    - **Filename:** `../dataset/audio/<species>/analysis/bvp_ids_<YYYYMMDD-HHMMSS>.csv`
    - **Description:** A CSV file containing the identifiers for each BVP (Best Vocalization Period) that was processed and included in the feature set. The columns include: `src_file`, `segment_num`, `cluster_feat_start`, `cluster_feat_end`, `coe_idx`, `pad_before`, and `pad_after`.

2.  **Cluster Features File**
    - **Filename:** `../dataset/audio/<species>/analysis/feats_<n_mels>x<n_frms>_<YYYYMMDD-HHMMSS>.csv`
    - **Description:** A CSV file containing the extracted feature vectors. Each row corresponds to a BVP listed in the BVP IDs file, and the values are the flattened spectrogram data, normalized to unity peak magnitude.
```