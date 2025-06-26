"""
vade_post.py

This script post-processes summary and anomal files from vade_clean 
to arrive at aggregated discard recommendations as follows:
* model selection criteria are applied to the model training results to determine 
which trained iteration models to use anomalousness scores from
* aggregate anomalousness scores from models that pass selection criteria and make 
discard recommendations

06-24-2025
- minor changes to move execution path down one level in folder restructuring for github

"""

import os
import pathlib
import math
import time
from datetime import datetime
import glob
import numpy as np
import pandas as pd
import scipy.optimize as opt

import argparse

"""
function: below_thresh

Determines the difference between the number of elements in the input vector, x, that are below the
input threshold, thresh, and the target number of such values, sum_target.

"""
def below_thresh(thresh, x, sum_target):
    return np.count_nonzero(x<thresh) - sum_target
    

def main():
    parser = argparse.ArgumentParser(
        description='Post-processes summary and anomal files from vade_clean to arrive at aggregated discard recommendations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to post-process VADE outputs for',
                        type=str)
    parser.add_argument('--hi-size-thresh', '-s',
                        help='Learning Rate.',
                        type=float, default=0.2)
    parser.add_argument('--hi-loss-thresh', '-l',
                        help='Learning Rate.',
                        type=float, default=0.2)

    species = args.species
    hi_loss_thresh = args.hi_loss_thresh
    hi_size_thresh = args.hi_size_thresh

    print('\n ************************** Inputs ****************************\n')
    print('bird species:',species,'\n hi_loss_thresh=',hi_loss_thresh,'\n hi_size_thresh=',hi_size_thresh)

    ### Input files
    src_dir = pathlib.Path('../dataset/audio')
    src_folder = os.path.join(src_dir, species)
    sum_files_list = glob.glob(
        os.path.join(src_dir, species, 'analysis', 'artifacts', 'summary_c'+str(n_clusters)+'_*'+'.csv')) 
    sum_file = max(sum_files_list, key=os.path.getmtime)
    
    # read the vade iterations summary from file
    try:
        df_sum = pd.read_csv(sum_file)
    except Exception as err:
        print('File read err for ', sum_file, ':', err)
        
    # read anomal file corresponding to summary file    
    anomal_file = sum_file.replace('summary_','anomal_')
    # read the vade iterations anomalousness scores from file
    try:
        df_anomal = pd.read_csv(anomal_file)
    except Exception as err:
        print('File read err for ', anomal_file, ':', err)
    
    ### clean-up summary file
    df_sum = df_sum.dropna()
    """ 
    model selection
    vade model exclusion criteria
    * vade training or test loss greater than the median value for all runs by more than 
    hi_loss_thresh percent
    * largest vade model cluster size greater than the median for all runs by more than 
    hi_size_thresh percent
    """
    ex_labels = []
    train_med = np.median(df_sum.train_loss)
    test_med = np.median(df_sum.test_loss)
    csize0_med = np.median(df_sum.vade_size_0)
    #print('median values: train_loss={:.2f}, test_loss={:.2f}, vade_size_0={:.2f}'\
    #      .format(train_med, test_med, csize0_med))

    df_ex = df_sum[(df_sum.train_loss>train_med*(1+hi_loss_thresh)) \
                   | (df_sum.test_loss>test_med*(1+hi_loss_thresh)) \
                   | (df_sum.vade_size_0>csize0_med*(1+hi_size_thresh))]
    print('number of exclustions={}'.format(len(df_ex.index)))    
    print('iteration labels excluded:', df_ex['Unnamed: 0'].to_list())
    """
    # determine threshold for each iteration
    threshs = {}
    target_pct = 0.1
    sum_target = np.round(len(df_anomal.index)*target_pct)
    for col in df_anomal.columns.to_list()[2:]:
        result = opt.root_scalar(below_thresh, args=(df_anomal[col], sum_target), bracket=(0,1))
        threshs[col] = result.root
        
    """
main()