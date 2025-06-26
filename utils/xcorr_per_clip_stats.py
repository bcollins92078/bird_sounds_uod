'''
xcorr_median_per_clip_avg.py
This python script computes the per-clip average cross-correlation (xcorr) from the previously 
saved xcorr table and computes "description" stats for these pairwise, per-clip averages:
    count
    mean 
    std 
    min_val 
    q1 
    median 
    q3 
    max_val 
    
06-25-2025
- minor changes to move execution path down one level as part of folder restructuring for phase 2

01-30-2025
- fixed bug that resulted in divide by zero when there is exactly one clip in a meta_type category
'''
import os
import pathlib
import glob
import pandas as pd
import numpy as np
import time
import argparse
import sys

'''
function: per_clip_xcorr
This function computes the per sample average cross-correlation across all other samples

Inputs:
    - xcoor - pairwise cross-correlation matrix

Outputs:
    - descriptive statistics
    
'''
def per_clip_xcorr(xcorr):
    descript_stats = {'count':-1.0, 'mean':-1.0, 'std':-1.0, 'min_val':-1.0, 'q1':-1.0, 'median':-1.0, 'q3':-1.0, 'max_val':-1.0}

    num_samps = xcorr.shape[0]
    per_samp_xcorr = np.zeros(num_samps)
    for x in range(num_samps):
        sum_xcorr = 0
        for j in range(x+1,num_samps):
            sum_xcorr += xcorr[x,j]

        for i in range(0,x):
            sum_xcorr += xcorr[i,x]
            
        per_samp_xcorr[x] = sum_xcorr/(num_samps-1)

    # descriptive statistics
    descript_stats['count'] = per_samp_xcorr.size
    descript_stats['mean'] = np.mean(per_samp_xcorr)
    descript_stats['std'] = np.std(per_samp_xcorr)
    descript_stats['min_val'] = np.min(per_samp_xcorr)
    descript_stats['q1'] = np.percentile(per_samp_xcorr, 25)
    xcorr_median = np.median(per_samp_xcorr)
    descript_stats['median'] = xcorr_median
    descript_stats['q3'] = np.percentile(per_samp_xcorr, 75)
    descript_stats['max_val'] = np.max(per_samp_xcorr)
    #descript_stats['1-median'] = 1-xcorr_median
    
    return descript_stats
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
    
def main():
    parser = argparse.ArgumentParser(
        description='computes the per-clip average cross-correlation (xcorr) from the previously saved xcorr table and outputs description stats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to perform processing for',
                        type=str)
    parser.add_argument('--filename',
                        help='name of .csv file containing cross-correlation table',
                        type=str,default='xcorr_*.csv')
    parser.add_argument('--song-call', '-s',
                        help='does species have distinct song and call (1=yes), 0=no)',
                        type=int, default=1)
    
    args = parser.parse_args()
    species = args.species
    filename = args.filename
    song_call = args.song_call
    
    src_dir = pathlib.Path('../dataset/audio')
    species_folder = os.path.join(src_dir, species)

    if filename == 'xcorr_*.csv':     # if filename is the default (i.e., not specified)
        # get list of all files with name matching 'xcorr_*.csv'
        files_list = glob.glob(os.path.join(species_folder, 'analysis', 'xcorr_*.csv'))
        if len(files_list) == 0:
            print('\nNo xcorr tables files found for', species,'!!!')
            sys.exit()
        xcorr_file = max(files_list, key=os.path.getmtime)
    else:
        xcorr_file = os.path.join(species_folder, 'analysis', filename)
    
    try:
        xcorr = np.loadtxt(xcorr_file, delimiter=',')
    except Exception as err:
        print('Error loading xcorr file:', xcorr_file, err)
        sys.exit()
        
    print('************************** xcorr_per_clip_stats.py inputs ****************************')    
    print('bird species:', species)
    print('xcorr file:', xcorr_file, ':', xcorr.shape)    

    # load src_file and BVP details from the latest bvp_ids file 
    # get list of all BVP IDs files in analysis folder and select the latest one
    files_list = glob.glob(os.path.join(species_folder, 'analysis', 'bvp_ids_*.csv')) 
    ids_file = max(files_list, key=os.path.getmtime)
    if ids_file == '':
        print('No bvp_ids*.csv file found')
        sys.exit()
    
    df_ids = pd.read_csv(ids_file)
    print('BVP IDs file:', ids_file, ':', df_ids.shape)
    print('song/call flag: ', song_call)
    print('\n*************************** xcorr_per_clip_stats.py outputs ****************************')
    if song_call==1 and 'meta_type' not in df_ids.columns:
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

        df_ids.to_csv(ids_file, index=False)

    ### compute per sample average xcorr (across all other samples)
    print('*** OVERALL Stats ***')
    descript = per_clip_xcorr(xcorr)
    
    print_descript(descript)
    print(f"median complexity score = {(1-descript['median']):0.4f}")

    if song_call==1:
        '''
        Extract xcorr values for song and call labeled clips and compute a 
        song/call/other weighted average xcor value
        '''
        df_calls = df_ids[df_ids.meta_type == 'call']
        df_songs = df_ids[df_ids.meta_type == 'song']
        df_others = df_ids[df_ids.meta_type == 'other']
        num_calls = len(df_calls.index)
        num_songs = len(df_songs.index)
        num_others = len(df_others.index)
        weighted_complx = 0
        
        # extract normalized peak xcorr array for just the call labeled clips
        if num_calls > 1:
            print('\n*** CALL label stats ***')
            call_xcorr = np.zeros((num_calls,num_calls))
            call_idx_list = df_calls.index.tolist()
            sum_xcorr = 0
            cnt_xcorr = 0
            for i in range(num_calls):
                 for j in range(i+1,num_calls):
                        call_xcorr[i,j] = xcorr[call_idx_list[i],call_idx_list[j]]
                        sum_xcorr += xcorr[call_idx_list[i],call_idx_list[j]]
                        cnt_xcorr += 1
                        
            descript = per_clip_xcorr(call_xcorr)        
            print_descript(descript)
            call_complx = 1-descript['median']
            print(f"median call complexity score = {call_complx:0.4f}")
            weighted_complx += call_complx*num_calls
        else:   # take care of label count of exactly one corner-case (pairwise count = 0)
            num_calls = 0
        
        # extract normalized peak xcorr array for just the song labeled clips
        if num_songs > 1:
            print('\n*** SONG label stats ***')
            song_xcorr = np.zeros((num_songs,num_songs))
            song_idx_list = df_songs.index.tolist()
            sum_xcorr = 0
            cnt_xcorr = 0
            for i in range(num_songs):
                 for j in range(i+1,num_songs):
                        song_xcorr[i,j] = xcorr[song_idx_list[i],song_idx_list[j]]
                        sum_xcorr += xcorr[song_idx_list[i],song_idx_list[j]]
                        cnt_xcorr += 1
                        
            descript = per_clip_xcorr(song_xcorr)        
            print_descript(descript)
            song_complx = 1-descript['median']
            print(f"median song complexity score = {song_complx:0.4f}")
            weighted_complx += song_complx*num_songs
        else:   # take care of label count of exactly one corner-case (pairwise count = 0)
            num_songs = 0
        
        # extract normalized peak xcorr array for just the other labeled clips
        idx = 0
        if num_others > 1:
            print('\n*** OTHER label stats ***')
            other_xcorr = np.zeros((num_others,num_others))
            other_idx_list = df_others.index.tolist()
            sum_xcorr = 0
            cnt_xcorr = 0
            for i in range(num_others):
                 for j in range(i+1,num_others):
                        other_xcorr[i,j] = xcorr[other_idx_list[i],other_idx_list[j]]
                        sum_xcorr += xcorr[other_idx_list[i],other_idx_list[j]]
                        cnt_xcorr += 1
                        
            descript = per_clip_xcorr(other_xcorr)        
            print_descript(descript)
            other_complx = 1-descript['median']
            print(f"median other complexity score = {other_complx:0.4f}")
            weighted_complx += other_complx*num_others
        else:   # take care of label count of exactly one corner-case (pairwise count = 0)
            num_other = 0
            
        # compute label weighted xcorr
        num_labels = num_calls + num_songs + num_others
        weighted_complx = weighted_complx / num_labels
        print(f"\nweighted median complexity score = {weighted_complx:0.4f}")
    
main()        