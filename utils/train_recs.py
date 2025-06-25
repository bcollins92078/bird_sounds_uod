"""
train_recs.py

This python script processes the discard recommendations in discard_*.csv and save 
per-sample recommendations whether to include the sample in classifier training in 
in the species _bvp*.csv

*   per-sample discard recommendations from both VADE and VAE pretraining models are 
    included in the discard_*.csv files so the code has to aggregate across these 
    two model types (VADE, VAE, AND, OR)
*   the primary motivation for performing this function in a separate script (as opposed 
    to just adding it to vade_post.py) is that often the number of VADE models that pass 
    the selection criteria (a test implemented in vade_post.py) is too small to proceed so 
    multiple runs of vade_clean.py and manual combining of results is sometimes required

06-25-2025
* minor changes to move execution path down one level as part of folder restructuring for phase 2

08-30-2023:
* output summary statistics

07-03-2024:
* added z_dim argument
"""

import sys
import argparse
import os
import pathlib
import math
import time
from datetime import datetime
import glob
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description='Post-processes discard_c<cluster_num>*.csv files from vade_post.py to arrive at per-sample recommendations whether to include in classifier training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to post-process outputs for',
                        type=str)
    parser.add_argument('--n-clusters', '-c',
                        help='Number of classes.',
                        type=int, default=2)
    parser.add_argument('--z-dim', '-z',
                        help='Number of latent dimensions.',
                        type=int, default=10)
    parser.add_argument('--aggregate', '-a',
                        help='aggregation option (vae, vade, and, or).',
                        type=str)

    # Run parameters
    args = parser.parse_args()
    species = args.species
    n_clusters = args.n_clusters
    z_dim = args.z_dim
    agg = args.aggregate
    src_dir = pathlib.Path('../dataset/audio')

    print('************************** train_recs.py: Inputs ****************************')
    print('bird species:',species,'\n number of clusters=',n_clusters,'\n latent dimensions=',z_dim)
    print('\n aggregation option=',agg)
    
    # get list of all discard files in artifacts folder and select the latest one
    src_folder = os.path.join(src_dir, species)
    disc_files_list = glob.glob(
        os.path.join(src_dir, species, 'analysis', 'artifacts', 'discards_fix_c'+str(n_clusters)+'_z'+str(z_dim)+'_*'+'.csv')) 
    disc_file = max(disc_files_list, key=os.path.getmtime)

    print('Discards File:',disc_file)
    
    # read the vade discard recommendations from file
    try:
        df_disc = pd.read_csv(disc_file)
    except Exception as err:
        print('File read err for ', disc_file, ':', err)
        sys.exit('Error Exit!')
        
    # get list of <species>_bvp*.csv files in analysis folder and select the latest one
    bvp_files_list = glob.glob(
        os.path.join(src_dir, species, 'analysis', species + '_bvp*.csv')) 
    bvp_file = max(bvp_files_list, key=os.path.getmtime)

    print('Input BVP File:',bvp_file)
    
    # read the _bvp file
    try:
        df_bvp = pd.read_csv(bvp_file)
    except Exception as err:
        print('File read err for ', bvp_file, ':', err)
        sys.exit('Error Exit!')

    """
    discards_*.csv files contain individual model (both vade and pretraining) discard 
    recommendations and separate columns for vade and pretraining tallies of these 
    recommendations. The processing here is to ...

    *   combine the aggregated vade and pretraining model recommendations to arrive at a final 
        per-sample discard recommendation based on the aggregation method specified
    """
    
    # compute final discard recommendation based on aggregation of vade models majority vote and that if pretrain models
    if agg=='or':
        df_disc['discard'] = df_disc['vade_maj'] | df_disc['pre_maj']
    elif agg=='and':
        df_disc['discard'] = df_disc['vade_maj'] & df_disc['pre_maj']
    elif agg=='vae':
        df_disc['discard'] = df_disc['pre_maj']
    elif agg=='vade':
        df_disc['discard'] = df_disc['vade_maj']
    else:
        print('Invalid aggregation option:',agg)
        sys.exit('Error Exit!')
        
    df_disc['discard'].value_counts()
    
    df_bvp['train_samp'] = ''
    for row in df_disc.itertuples():
        if row.discard:
            df_bvp.loc[(df_bvp.src_file==row.src_file) & (df_bvp.segment_num==row.segment_num),'train_samp'] = False
        else:
            df_bvp.loc[(df_bvp.src_file==row.src_file) & (df_bvp.segment_num==row.segment_num),'train_samp'] = True
            
    #print(df_bvp['train_samp'].value_counts())
    
    # write df_bvp out to a csv file with a new timestamp in the name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    new_bvp_file = os.path.join(src_dir, species, 'analysis',species + '_bvp_'+timestamp+'.csv')
    df_bvp.to_csv(new_bvp_file,index=False)
    
    """ 
    Output summary statistics:
    * num discards from vade ensemble and vae ensemble and the discards count after aggregation
    * correlation between vade and vae
    """
    
    print('\n************************** train_recs.py: Outputs ****************************')
    print('New BVP File:',new_bvp_file)
    print('Total Recordings:',df_bvp.src_file.nunique())
    print('Total BVPs:',len(df_bvp.index),'(',\
            len(df_bvp[df_bvp.meta_type=='song'].index),'songs,',\
            len(df_bvp[df_bvp.meta_type=='call'].index),'calls,',\
            len(df_bvp[df_bvp.meta_type=='both'].index),'both,',\
            len(df_bvp[df_bvp.meta_type=='other'].index),'other )')
    df_postpre = df_bvp[df_bvp.train_samp != '']
    num_postpre = len(df_postpre.index)
    print('BVPs after preprocessing:',num_postpre,'(',\
            len(df_postpre[df_postpre.meta_type=='song'].index),'songs',\
            len(df_postpre[df_postpre.meta_type=='call'].index),'calls',\
            len(df_postpre[df_postpre.meta_type=='both'].index),'both',\
            len(df_postpre[df_postpre.meta_type=='other'].index),'other )')
    print('Total Discards:',len(df_postpre[df_postpre.train_samp==False].index),'(',\
            '{:.1f}%'.format(100*len(df_postpre[df_postpre.train_samp==False].index)/num_postpre),')')
    print('Net Training Samples',len(df_postpre[df_postpre.train_samp==True].index))
main()    