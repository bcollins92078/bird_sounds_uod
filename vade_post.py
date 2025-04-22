"""
vade_post.py

This script post-processes summary and anomal files from vade_clean 
to arrive at aggregated discard recommendations as follows:
* model selection criteria are applied to training results from both VADE and VAE pretraining 
models to determine which models to use anomalousness scores from
* aggregate anomalousness scores from VADE and VAE pretraining models that pass selection 
criteria and make discard recommendations

09-30-2024
- added commandline option to disable cluster0 size and training/test loss selection criteria

03-29-2024: 
- minor output text change - corrected "vade_post_fix.py" to "vade_post.py"

03-12-2024:
- added computation of mean homogeneity_score across selected models if species has both song and call

03-05-2024:
- output "discards_*" filename changed to "discards_fix_*" to lineup with vade_post_fix.py
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
import warnings
warnings.filterwarnings('ignore')

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
                        help='model largest cluster size fraction above median selection criteria (-1 disables)',
                        type=float, default=0.2)
    parser.add_argument('--hi-loss-thresh', '-l',
                        help='Training/test loss fraction above median selection criteria (-1 disables)',
                        type=float, default=0.2)
    parser.add_argument('--n-clusters', '-c',
                        help='Number of classes.',
                        type=int, default=2)
    parser.add_argument('--z-dim', '-z',
                        help='Number of latent dimensions.',
                        type=int, default=10)
    parser.add_argument('--max-discard-frac', '-d',
                        help='Max fraction of samples to discard.',
                        type=float, default=0.1)
    parser.add_argument('--degen-pct-size-thresh', '-b',
                        help='degenerate model largest cluster absolute percentage size threshold to exclude model.',
                        type=float, default=0.85)

    args = parser.parse_args()
    species = args.species
    hi_loss_thresh = args.hi_loss_thresh
    hi_size_thresh = args.hi_size_thresh
    n_clusters = args.n_clusters
    z_dim = args.z_dim
    if n_clusters > 1:
        degen_pct_size_thresh = args.degen_pct_size_thresh
    else: # in the n_clusters = 1 case all cluster_size0 == 1 and this parameter doesn't apply
        degen_pct_size_thresh = 1
        
    print('************************** c{}: vade_post.py inputs ****************************'.format(n_clusters))
    print('bird species:',species,'\n n_clusters=',n_clusters,'\n latent_dim=', z_dim )
    print(' degen_pct_size_thresh=',degen_pct_size_thresh,'\n max_discard_frac=',args.max_discard_frac)
    print(' hi_loss_thresh=',hi_loss_thresh,'\n hi_size_thresh=',hi_size_thresh)
    print('\n*************************** c{}: vade_post.py outputs ****************************'.format(n_clusters))
    
    ### Input files
    src_dir = pathlib.Path('dataset/audio')
    src_folder = os.path.join(src_dir, species)
    sum_files_list = glob.glob(
        os.path.join(src_dir, species, 'analysis', 'artifacts', 'summary_c'+str(n_clusters)+'_z'+str(z_dim)+'*'+'.csv')) 
        #os.path.join(src_dir, species, 'analysis', 'artifacts', 'summary_*'+'.csv')) 
    sum_file = max(sum_files_list, key=os.path.getmtime)
    
    # read the vade iterations summary from file
    try:
        df_sum = pd.read_csv(sum_file,index_col=0)
    except Exception as err:
        print('File read err for ', sum_file, ':', err)
        sys.exit()
        
    # read anomal file corresponding to summary file    
    anomal_file = sum_file.replace('summary_','anomal_')
    # read the vade iterations anomalousness scores from file
    try:
        df_anomal = pd.read_csv(anomal_file)
    except Exception as err:
        print('File read err for ', anomal_file, ':', err)
        sys.exit()
    
    # get summary of _all_ pretrain models before dropping failed vade training attempts    
    pre_csize0_med = np.median(df_sum[df_sum.pre_size_0<degen_pct_size_thresh].pre_size_0)
    if hi_size_thresh > 0:    # if cluster0 size selection criteria is not disabled
        df_pre_ex = df_sum[(df_sum.pre_size_0>pre_csize0_med*(1+hi_size_thresh)) \
                            | (df_sum.pre_size_0>degen_pct_size_thresh)]
    else:    # if cluster0 size selection criteria is disabled
        df_pre_ex = df_sum[df_sum.pre_size_0>degen_pct_size_thresh]
        
    
    ### clean-up summary file
    df_sum = df_sum.dropna()
    """ 
    model selection
    vade model exclusion criteria
    * vade training or test loss greater than the median value for all runs by more than 
    hi_loss_thresh percent
    * largest vade model cluster size greater than the median for all runs by more than 
    hi_size_thresh percent OR if the absolute percentage size of this largest cluster size is 
    greater than the degen_pct_size_thresh (this second criteria doesn't apply in n_clusters = 1 case)
    
    """
    train_med = np.median(df_sum.train_loss)
    test_med = np.median(df_sum.test_loss)
    csize0_med = np.median(df_sum.vade_size_0)
    
    #print(train_med, test_med, csize0_med, pre_csize0_med)
    
    # vade models excluded due to training or test loss (if not disabled)
    if hi_loss_thresh > 0:
        df_vade_ex_train = df_sum[df_sum.train_loss>train_med*(1+hi_loss_thresh)]
        df_vade_ex_test = df_sum[df_sum.test_loss>test_med*(1+hi_loss_thresh)]
    else:    # if training/test loss model selection _is_ disabled
        df_vade_ex_train = pd.DataFrame(columns=df_sum.columns)
        df_vade_ex_test = pd.DataFrame(columns=df_sum.columns)
        
    # vade models excluded due to cluster0 size (if not disabled)
    if hi_size_thresh > 0:
        df_vade_ex_size = df_sum[(df_sum.vade_size_0>csize0_med*(1+hi_size_thresh)) \
                                 | (df_sum.vade_size_0>degen_pct_size_thresh)]
    else:    # if cluster0 size model selection _is_ disabled
        df_vade_ex_size = pd.DataFrame(columns=df_sum.columns)
    
    #print(csize0_med*(1+hi_size_thresh),degen_pct_size_thresh)
    print('number of pretrain exclustions={}'.format(len(df_pre_ex.index)))
    pre_ex_labels = df_pre_ex.index.to_list()
    print('pretrain models excluded for cluster0 size:', pre_ex_labels)
    
    print('number of VADE exclustions={}'.format(len(df_vade_ex_train.index) + \
                                                len(df_vade_ex_test.index) + \
                                                len(df_vade_ex_size.index)))
    vade_ex_train = df_vade_ex_train.index.to_list()
    vade_ex_test = df_vade_ex_test.index.to_list()
    vade_ex_size = df_vade_ex_size.index.to_list()
    print('vade models excluded for training loss:', vade_ex_train)
    print('vade models excluded for test loss:', vade_ex_test)
    print('vade models excluded for cluster0 size:', vade_ex_size)

    """ 
    Process anomal file

    After model selection identifies the iterations that are to be excluded...
    * load the anomal_c*.csv file that matches the summary file
    * drop the columns from the anomalousness scores file identified by both pretrain and vade
    model selection
    * determine the threshold value to be applied to each remaining vade column to achieve the 
    desired number of discards (accommodating preclean 0 values)
    * mark each sample with a discard recommendation
    * tallies the discard recommendations across all iterations in the file
    * outputs the resulting anomal file
    """

    # drop columns identified by model selection
    df_anomal.drop(labels=vade_ex_train+vade_ex_test+vade_ex_size,axis='columns',inplace=True)
    
    # drop pretraining discard recommendation columns from following vade post-processing
    df_disc = df_anomal[df_anomal.columns.drop(list(df_anomal.filter(regex='pre_')))]
    
    # determine threshold for each remaining iteration
    raw_cols = df_disc.columns.to_list()[2:] # remember list of these raw results columns
    threshs = {}
    sum_target = np.round(len(df_disc.index)*args.max_discard_frac)
    for col in raw_cols:
        try:
            result = opt.root_scalar(below_thresh, args=(df_disc[col], sum_target), bracket=(0,1))
        except Exception as err:
            print(col, err)
        #result = opt.root_scalar(below_thresh, args=(df_disc[df_disc[col]>0][col], sum_target), bracket=(0,1))
        threshs[col] = result.root
    
    # apply threshold to get discard recommendations for each iteration's results
    for col in threshs:
        df_disc.loc[:,'vade'+col[-17:]] = df_anomal[col]<threshs[col]
    
    # add tally column that sums the number of true values
    df_disc['vade_tally'] = df_disc.filter(like='vade_',axis=1).sum(axis=1)

    # post process pre-training discard recommendations to df_disc
    df_pre = pd.DataFrame()
    for col in [label for label in df_anomal.columns if 'pre_' in label]:
        df_pre[col] = df_anomal[col]
        df_disc[col] = df_anomal[col]
    
    col_labels = []
    for label in pre_ex_labels:
        col_labels.append('pre'+label[-17:])
    df_pre.drop(labels=col_labels,axis='columns',inplace=True)
    df_disc.drop(labels=col_labels,axis='columns',inplace=True)
    df_disc['pre_tally'] = df_pre.sum(axis=1)
    
    # drop raw_cols columns from df_disc
    df_disc.drop(columns=raw_cols, inplace=True)
    
    num_pre = len(df_disc.filter(like='pre_',axis=1).columns)-1
    print('number of pretrain models selected: {}'.format(num_pre))
    num_vade = len(df_disc.filter(like='vade_',axis=1).columns)-1
    print('number of VADE models selected: {}'.format(num_vade))
    # compute the majority for vade and pretraining models
    pre_maj = round(num_pre/2 + 0.5000000001)
    vade_maj = round(num_vade/2 + 0.5000000001)

    # majority vote across vade models
    df_disc['vade_maj'] = (df_disc['vade_tally'] >= vade_maj)

    # majority vote across pretraining vae models
    df_disc['pre_maj'] = (df_disc['pre_tally'] >= pre_maj)

    num_samp = len(df_disc.index)
    print('number of samples after preprocessing: {}; max discards: {}'.format(num_samp,sum_target))
    cnt_pre = df_disc.pre_maj.sum()
    print('number of pretrain models majority discards: {} ({:.2}%)'.format(cnt_pre,100*cnt_pre/num_samp))
    cnt_vade = df_disc.vade_maj.sum()
    print('number of vade models majority discards: {} ({:2.2}%)'.format(cnt_vade,100*cnt_vade/num_samp))
    num_or = len(df_disc[df_disc['vade_maj'] | df_disc['pre_maj']].index)
    print('number of pretrain OR vade majority discards: {} ({:2.2}%)'.format(num_or,100*num_or/num_samp))
    num_and = len(df_disc[df_disc['vade_maj'] & df_disc['pre_maj']].index)
    print('number of pretrain AND vade majority discards: {} ({:2.2}%)'.format(num_and,100*num_and/num_samp))
    correl = df_disc[['pre_maj','vade_maj']].corr().loc['pre_maj','vade_maj']
    print('correlation between pretrain and vade majority votes: {:.2}'.format(correl))
    # write discard recommendations (both from pretraining and vade models) to discards file 
    df_disc.to_csv(anomal_file.replace('anomal_','discards_fix_'), index=False)
    
    # compute and print the mean homogeneity scores in the summary for the selected models if present
    if 'homo_met' in df_sum.columns:
        df_sum.drop(labels=vade_ex_train+vade_ex_test+vade_ex_size,axis='index',inplace=True)
        print('average homogeneity_score: {:.3}'.format(df_sum.homo_met.mean()))
        
main()