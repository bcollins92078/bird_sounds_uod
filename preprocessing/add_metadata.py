"""
add_metadata.py

This script adds metadata from local_birds_meta.csv to the specified species' _bvp.csv file

Input:
* bird species common name

Output:
* updated <bird_species>_bvp_<timestamp>.csv file

06-25-2025
- minor changes to move execution path down one level as part of folder restructuring for phase 2

"""

import sys
from sys import argv
import os
import glob
import pathlib
import math
import time
from datetime import datetime
import numpy as np
import pandas as pd
import csv

import warnings
warnings.filterwarnings('ignore')

"""
function: main

Read the latest <bird_species>_bvp.csv file into a dataframe
Lookup each src_file in <bird_species>_bvp.csv in local_birds_meta.csv and get the meta_type field 
Add a column to <bird_species>_bvp dataframe
Write the updated <bird_species>_bvp dataframe to <bird_species>_bvp_<timestamp>.csv
"""
def main(argv):
    base_path = '../dataset/audio'
    
    # Assess the inputs
    if len(argv) != 2:
        print('format: add_metadata.py <bird_species>')
        sys.exit()
        
    script, species = argv
    print('species:', species)
    
    src_folder = os.path.join(base_path, species)

    #### Setup parameters and load the <species_name>_bvp.csv file
    # get list of all _bvp files in analysis folder and select the latest one
    bvp_files_list = glob.glob(os.path.join(src_folder, 'analysis', species + '_bvp*.csv')) 
    bvp_file = max(bvp_files_list, key=os.path.getctime)
    if bvp_file == '':
        print('No _bvp*.csv file found')
        sys.exit()
        
    # load bvp file
    try:
        df_bvp = pd.read_csv(bvp_file)
        print('adding metadata to ', bvp_file)
    except Exception as err:
        print('File read err for ', bvp_file, ':', err)
        sys.exit()

    # Read local_birds_meta.csv for per audio file average SINRs 
    df_meta = pd.read_csv('local_birds_meta.csv')
    df_meta.set_index('ID',inplace=True)
    
    # add column for vocalization type to _bvp dataframe
    df_bvp['meta_type'] = ''
    
    # copy Rec_Content from metadata file to df_bvp
    for row in df_bvp.itertuples():
        df_bvp.at[row.Index,'meta_type'] = df_meta.at[int(row.src_file.replace('.mp3', '')), 'Rec_Content']
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M-%S")
    
    # write src_file and segment_num data (IDs) to a .csv
    bvp_file = os.path.join(src_folder, 'analysis', species+'_bvp_'+timestamp+'.csv')
    df_bvp.to_csv(bvp_file, index=False)
main(argv)