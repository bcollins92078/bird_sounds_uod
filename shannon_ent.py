'''
shannon_ent.py
This file contains a python script that computes descriptive statistics over the per clip
Shannon entropy

01-31-2025
- Adding code to compute average shannon entropy for each meta_type label
'''

import time
import numpy as np
import pandas as pd
import os
import pathlib
import glob
import argparse
import sys

'''
function: shannon_ent
- Normalize the input values (spectrogram) so that they sum to one. This ensures that you have a 
valid probability distribution
- Treat the normalized values as a probability distribution
- Compute Shannon entropy
'''
def shannon_ent(feat):
    normed_feat = feat/np.sum(feat)
    return -np.sum(normed_feat * np.log2(normed_feat))
    
    
'''
Function: print_descript
This function prints descriptive stats in a nice format
'''
def print_descript(descript):
    print(f"count = {descript['count']}")
    print(f"mean = {descript['mean']:0.4f}")
    print(f"std = {descript['std']:0.4f}")
    print(f"min_val = {descript['min_val']:0.4f}")
    print(f"Q1 = {descript['q1']:0.4f}")
    print(f"median = {descript['median']:0.4f}")
    print(f"Q3 = {descript['q3']:0.4f}")
    print(f"max_val = {descript['max_val']:0.4f}")
    return


'''
main loop
    Loop thru the features and compute the entropy of each row collecting them into an array. 
    Then compute and output the descriptive stats
'''
def main():
    parser = argparse.ArgumentParser(
        description='computes the per-clip statistics of Shannon entropy for specified bird species',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to perform processing for',
                        type=str)
    args = parser.parse_args()
    species = args.species
    src_dir = pathlib.Path('dataset/audio')
    species_folder = os.path.join(src_dir, species)

    # get list of all 32x40 features files in analysis folder and select the latest one
    files_list = glob.glob(os.path.join(species_folder, 'analysis', 'feats_32x40*.csv'))
    if len(files_list) == 0:
        print('\nNo 32x40 features files found for', species,'!!!')
        sys.exit()
        
    feats_file = max(files_list, key=os.path.getmtime)
    try:
        feats = np.loadtxt(feats_file, delimiter=',')
        print('Processing features file:', feats_file)
    except Exception as err:
        print('Error opening feats file:', feats_file, err)
        sys.exit()

    # load src_file and BVP details from the bvp_ids file that match feats_file
    bvp_ids_file = feats_file.replace('feats_32x40','bvp_ids')
    df_ids = pd.read_csv(bvp_ids_file)
    print('BVP IDs file:', bvp_ids_file, ':', df_ids.shape)
        
    if 'meta_type' not in df_ids.columns:
        # get list of all _bvp files in analysis folder and select the latest one
        files_list = glob.glob(os.path.join(species_folder, 'analysis', species + '_bvp*.csv')) 
        bvp_file = max(files_list, key=os.path.getmtime)
        if bvp_file == '':
            print('No _bvp*.csv file found')
            sys.exit()
            
        # load bvp file
        try:
            df_bvp = pd.read_csv(bvp_file)
            print('getting meta_type from ', bvp_file)
        except Exception as err:
            print('File read err for ', bvp_file, ':', err)
            sys.exit()
        
        # get labels for song/call/other summary stats
        df_ids['meta_type'] = ''
        for row in df_ids.itertuples():
            df_ids.at[row.Index,'meta_type'] = df_bvp[(df_bvp.src_file==row.src_file) 
                                                     & (df_bvp.segment_num==row.segment_num)].meta_type.values[0]        

        df_ids.to_csv(bvp_ids_file, index=False)

    '''
    compute the overall mean shannon entropy
    '''
    print('\n*** OVERALL entropy stats ***')
    descript_stats = {'count':-1.0, 'mean':-1.0, 'std':-1.0, 'min_val':-1.0, 'q1':-1.0, 'median':-1.0, 'q3':-1.0, 'max_val':-1.0}
    num_samps = feats.shape[0]
    per_samp_ents = np.zeros(num_samps)
    #for i in range(num_samps):
    #    per_samp_ents[i] = shannon_ent(feats[i,:])
    for i in range(num_samps):
        ent = shannon_ent(feats[i,:])
        if np.isnan(ent):
            print('NaN encountered:', species, i)
        else:
            per_samp_ents[i] = ent

    # descriptive statistics
    descript_stats['count'] = per_samp_ents.size
    descript_stats['mean'] = np.mean(per_samp_ents)
    descript_stats['std'] = np.std(per_samp_ents)
    descript_stats['min_val'] = np.min(per_samp_ents)
    descript_stats['q1'] = np.percentile(per_samp_ents, 25)
    ent_median = np.median(per_samp_ents)
    descript_stats['median'] = ent_median
    descript_stats['q3'] = np.percentile(per_samp_ents, 75)
    descript_stats['max_val'] = np.max(per_samp_ents)

    print_descript(descript_stats)

    '''
    compute the mean shannon entropy for each of the call, song and other meta_type labels if non-zero members
    '''
    df_calls = df_ids[df_ids.meta_type == 'call']
    df_songs = df_ids[df_ids.meta_type == 'song']
    df_others = df_ids[df_ids.meta_type == 'other']
    num_calls = len(df_calls.index)
    num_songs = len(df_songs.index)
    num_others = len(df_others.index)

    if num_calls > 0:
        print('\n*** CALL label entropy stats ***')
        idx_list = df_calls.index.tolist()
        per_samp_ents = np.zeros(num_calls)
        cnt = 0
        for i in idx_list:
            per_samp_ents[cnt] = shannon_ent(feats[i,:])
            cnt += 1
        
        # descriptive statistics
        descript_stats['count'] = per_samp_ents.size
        descript_stats['mean'] = np.mean(per_samp_ents)
        descript_stats['std'] = np.std(per_samp_ents)
        descript_stats['min_val'] = np.min(per_samp_ents)
        descript_stats['q1'] = np.percentile(per_samp_ents, 25)
        ent_median = np.median(per_samp_ents)
        descript_stats['median'] = ent_median
        descript_stats['q3'] = np.percentile(per_samp_ents, 75)
        descript_stats['max_val'] = np.max(per_samp_ents)

        print_descript(descript_stats)
    
    if num_songs > 0:
        print('\n*** SONG label entropy stats ***')
        idx_list = df_songs.index.tolist()
        per_samp_ents = np.zeros(num_songs)
        cnt = 0
        for i in idx_list:
            per_samp_ents[cnt] = shannon_ent(feats[i,:])
            cnt += 1
        
        # descriptive statistics
        descript_stats['count'] = per_samp_ents.size
        descript_stats['mean'] = np.mean(per_samp_ents)
        descript_stats['std'] = np.std(per_samp_ents)
        descript_stats['min_val'] = np.min(per_samp_ents)
        descript_stats['q1'] = np.percentile(per_samp_ents, 25)
        ent_median = np.median(per_samp_ents)
        descript_stats['median'] = ent_median
        descript_stats['q3'] = np.percentile(per_samp_ents, 75)
        descript_stats['max_val'] = np.max(per_samp_ents)

        print_descript(descript_stats)
    
    if num_others > 0:
        print('\n*** OTHER label entropy stats ***')
        idx_list = df_others.index.tolist()
        per_samp_ents = np.zeros(num_others)
        cnt = 0
        for i in idx_list:
            per_samp_ents[cnt] = shannon_ent(feats[i,:])
            cnt += 1
        
        # descriptive statistics
        descript_stats['count'] = per_samp_ents.size
        descript_stats['mean'] = np.mean(per_samp_ents)
        descript_stats['std'] = np.std(per_samp_ents)
        descript_stats['min_val'] = np.min(per_samp_ents)
        descript_stats['q1'] = np.percentile(per_samp_ents, 25)
        ent_median = np.median(per_samp_ents)
        descript_stats['median'] = ent_median
        descript_stats['q3'] = np.percentile(per_samp_ents, 75)
        descript_stats['max_val'] = np.max(per_samp_ents)

        print_descript(descript_stats)
    
main()