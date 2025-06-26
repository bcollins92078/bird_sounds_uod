"""
plot_feat_32x40.py

Python script plots the feature vector from the latest feats_32x40.csv file for the specified 
species, file and segment.

06-25-2025
- minor changes to move execution path down one level as part of folder restructuring for phase 2
11-06-2024
- removed unused librosa imports
06-28-2024
- added header comment
"""
import sys
from sys import argv
import pathlib
import os
import glob
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import librosa
#import librosa.display
import csv

import warnings
warnings.filterwarnings('ignore')


def main(argv):
    base_path = os.path.join('../dataset', 'audio')

    # Assess the inputs
    if len(argv) != 4:
        print('format: plot_feat.py <bird_species> <filename> <segment_num>')
        sys.exit()
        
    script, bird_species, file, segment = argv
    print('bird_species:', bird_species, 'src_file:', file, 'segment_num:', segment)
    
    # get list of all 32x40 features files in analysis folder and select the latest one
    files_list = glob.glob(os.path.join(base_path, bird_species, 'analysis', 'feats_32x40*.csv'))
    feats_file = max(files_list, key=os.path.getmtime)
    try:
        feats = np.loadtxt(feats_file, delimiter=',')
    except Exception as err:
        print('Error opening feats file:', feats_file, err)

    # get list of all bvp_ids files in analysis folder and select the latest one
    bvp_ids_files_list = glob.glob(os.path.join(base_path, bird_species, 'analysis', 'bvp_ids*.csv'))
    bvp_ids_file = max(bvp_ids_files_list, key=os.path.getmtime)
    try:
        df_ids = pd.read_csv(bvp_ids_file)
    except Exception as err:
        print('Error opening bvp_ids file:', bvp_ids_file, err)
		
    # Do the src_file and segment_num specified on the command line appear in bvp_ids file
    segment_num = int(segment)
    if df_ids[(df_ids.src_file == file) & (df_ids.segment_num == segment_num)].empty == True:
        print('specified src_file, segment_num doesn\'t appear in bvp_ids file')
        sys.exit()
    
    # global constants (all-caps indicate that value needs to remain the same across scripts)
    MIN_FREQ = 1000
    NUM_MELS = 32
    SR = 22050
    FRM_SIZE = 1024
    NUM_FREQS = 32
    FMIN = 1000
    NUM_FRMS = 40
    
    # get index to row in feats that corresponds to specified src_file and segment_num
    idx = df_ids[(df_ids.src_file == file) & (df_ids.segment_num == segment_num)].index.astype(int)[0]
	
    # scale and reshape features for specified BVP
    flat_feat = np.zeros(NUM_MELS * NUM_FRMS)
    flat_feat[:] = feats[idx,:]/np.max(feats[idx,:])

    feat = np.reshape(flat_feat, (-1,NUM_MELS,NUM_FRMS,1), order='F')
	
    # plot 
    plt.figure(1)
    plt.imshow(feat[0][:,:,0], origin='lower')
    #plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")
	
main(argv)