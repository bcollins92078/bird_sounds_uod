"""
cvae_post.py

This script post-processes discard_cvae_*.csv files from vae_clean.py
to arrive at aggregated discard recommendations and record them in the sepecies _bvp.csv file:
* tally per-sample discard recommendations across the CVAE models and apply majority vote 
to aggregate into a final recommendation 
* record these per-sample recommendations in a species _bvp.csv file update

This is modeled after vade_post.py and specifically needed for species that VADE model training fails

10-14-2024
- renamed cvae_post.py (formerly vae_post.py) to be consistent with evolved naming convention
- added target discards commandline argument and "adjust" vote threshold to get as close to that
target as possible
- added writing of discard tallies, majority voting results and adjusted vote results back to 
input discards file
- removed recording of per-sample recommendations in the species _bvp file (to be performed in 
a separate utility like train_recs.py)

11-19-2024
- removed n_clusters commandline arg as it is not used
- noticed that some more of the recent changes were not logged (e.g., target discard rate)

05-02-2024
- added summary stats to match train_recs.py output
"""
import sys
import os
import pathlib
import math
import time
from datetime import datetime
import glob
import numpy as np
import pandas as pd

import argparse
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(
        description='Post-processes discard_cvae_*.csv file from cvae_clean to arrive at aggregated discard recommendations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to post-process CVAE outputs for',
                        type=str)
    parser.add_argument('--hi-loss-thresh', '-l',
                        help='Training/test loss fraction above median for all iterations to exclude a model.',
                        type=float, default=0.2)
    parser.add_argument('--max-discard-frac', '-d',
                        help='Max fraction of samples to discard.',
                        type=float, default=0.1)
    parser.add_argument('--target-discard-frac', '-t',
                        help='Target fraction of samples to discard.',
                        type=float, default=0.05)
    '''
    parser.add_argument('--n-clusters', '-c',
                        help='Number of classes.',
                        type=int, default=2)
    '''                    
    parser.add_argument('--z-dim', '-z',
                        help='Number of latent dimensions.',
                        type=int, default=10)

    args = parser.parse_args()
    species = args.species
    hi_loss_thresh = args.hi_loss_thresh
    max_discard_frac = args.max_discard_frac
    target_discard_frac = args.target_discard_frac
    #n_clusters = args.n_clusters
    z_dim = args.z_dim
    
    if target_discard_frac > max_discard_frac:
        print('target discards fraction ({}) must be less than max discards fraction ({})' 
        .format(target_discard_frac,max_discard_frac))
        sys.exit()
    
    print('************************** cvae_post.py Inputs ****************************')
    print('bird species:',species,'\n hi_loss_thresh=',hi_loss_thresh)
    print(' max discards fraction= ',max_discard_frac,'\n target discards fraction=',target_discard_frac)
    print('\n*************************** cvae_post.py Outputs ****************************')
    
    ### Input files
    src_dir = pathlib.Path('dataset/audio')
    src_folder = os.path.join(src_dir, species)
    sum_files_list = glob.glob(
        os.path.join(src_dir, species, 'analysis', 'artifacts', 'summary_cvae_a2_z'+str(z_dim)+'*'+'.csv')) 
    sum_file = max(sum_files_list, key=os.path.getmtime)
    
    # read the vae_clean iterations summary from file
    try:
        df_sum = pd.read_csv(sum_file)
    except Exception as err:
        print('File read err for ', sum_file, ':', err)
        sys.exit()
        
    # read discard file corresponding to summary file    
    discard_file = sum_file.replace('summary_','discard_')
    # read the vae iterations discard recommendations from file
    try:
        df_discard = pd.read_csv(discard_file)
    except Exception as err:
        print('File read err for ', discard_file, ':', err)
        sys.exit()
    print('discard file:', discard_file)
    
    """ 
    model selection
    vae model exclusion criteria
    * vae validation loss less than the median value for all runs by more than 
    hi_loss_thresh percent

    """
    val_med = np.median(df_sum.val_loss)

    df_ex = df_sum[df_sum.val_loss<val_med*(1+hi_loss_thresh)] 

    print('number of model exclustions={}'.format(len(df_ex.index)))
    ex_labels = df_ex['Unnamed: 0'].to_list()
    print('model iteration labels excluded:', ex_labels)
    
    df_discard.drop(labels=ex_labels,axis='columns',inplace=True)
    df_discard['tally'] = df_discard.filter(like='cvae_',axis=1).sum(axis=1)
    models_selected = len(df_discard.filter(like='cvae_',axis=1).columns)
    print('number of models selected: {}'.format(models_selected))
    
    '''
    # get list of <species>_bvp*.csv files in analysis folder and select the latest one
    bvp_files_list = glob.glob(
        os.path.join(src_dir, species, 'analysis', species + '_bvp*.csv')) 
    bvp_file = max(bvp_files_list, key=os.path.getmtime)

    # read the _bvp file
    try:
        df_bvp = pd.read_csv(bvp_file)
    except Exception as err:
        print('File read err for ', bvp_file, ':', err)
        sys.exit('Error Exit!')
    '''
    # compute the majority cvae models
    maj = round((len(df_discard.filter(like='cvae_',axis=1).columns)-1)/2 + 0.5000000001)
    #print('maj=',maj)
    
    # majority vote across models
    df_discard['maj'] = (df_discard['tally'] >= maj)
    maj_cnt = (df_discard['maj'] == True).sum()
    '''
    Adjust the vote threshold so that the number of discard recommendations matches the 
    target_discard_frac and add that to a "adjusted" column. The purposed of this is to 
    support fair comparisons between model ensembles
    '''
    num_samp = len(df_discard.index)
    target_discard_cnt = target_discard_frac*num_samp
    discard_frac = maj_cnt/num_samp
    last_vote_cnt = maj_cnt
    
    if discard_frac < target_discard_frac:
        # reduce vote_thres until adj_discard_cnt > target_discard_cnt
        adj_step = -1
        for adj_discard_thres in range(maj,1,-1):
            vote_cnt = (df_discard['tally'] > adj_discard_thres).sum()
            if vote_cnt >= target_discard_cnt:
                break
            else:
                last_vote_cnt = vote_cnt
        #print('discard_frac ({:.2}%) < target_discard_frac ({:.2}%)'.format(discard_frac,target_discard_frac), vote_cnt, last_vote_cnt)
        
    elif discard_frac > target_discard_frac:
        # increase vote_thres until adj_discard_cnt < target_discard_cnt
        adj_step = 1
        for adj_discard_thres in range(maj,models_selected):
            vote_cnt = (df_discard['tally'] > adj_discard_thres).sum()
            if vote_cnt <= target_discard_cnt:
                break
            else:
                last_vote_cnt = vote_cnt
        #print('discard_frac ({:.2}%) > target_discard_frac ({:.2}%)'.format(discard_frac,target_discard_frac), vote_cnt, last_vote_cnt)
    
    else: # unlikely exactly exactly equal case
        adj_discard_cnt = (df_discard['tally'] > maj).sum()
        adj_discard_thres = maj
        
    if np.absolute((vote_cnt/num_samp) - (target_discard_cnt/num_samp)) <= np.absolute((last_vote_cnt/num_samp) - (target_discard_cnt/num_samp)):
        adj_discard_cnt = vote_cnt
    else:
        adj_discard_cnt = last_vote_cnt
        adj_discard_thres -= adj_step
        
    #print(adj_discard_cnt, adj_discard_thres)
    #print(np.absolute((vote_cnt/num_samp) - (target_discard_cnt/num_samp)))
    #print(np.absolute((last_vote_cnt/num_samp) - (target_discard_cnt/num_samp)))
    
    # adjusted vote across models
    df_discard['adj'] = (df_discard['tally'] > adj_discard_thres)
    
    # write df_discard out to the discard_cvae_ file read above
    df_discard.to_csv(discard_file,index=False)    

    #print('\n************************** Summary Stats ****************************')
    max_discards = np.round(len(df_discard.index)*max_discard_frac)
    print('number of samples after preprocessing: {}; max discards: {}'.format(num_samp,max_discards))
    print('number of model majority vote discards: {} ({:.2}%)'.format(maj_cnt,100*discard_frac))
    print('number of adjusted threshold discards: {} ({:.2}%)'.format(adj_discard_cnt,100*adj_discard_cnt/num_samp))
main()