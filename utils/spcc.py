"""
spcc.py

 Python script computes pairwise cross correlations between the spectrogram clips comprising 
 one-species bird dataset (SPectrogram Cross-Correlation - SPCC), then takes the maximum value, 
 normalizes it and takes the average over all all pairwise values to compute a dataset 
 complexity metric as (1-spcc).
 
 06-25-2025
- minor changes to move execution path down one level as part of folder restructuring for phase 2

12-17-2024
 - added code to interpret vertical pad argument of zero to compute "full" correlation
 12-13-2024
 - minor edits
"""

import os
import pathlib
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlate2d
import time
from datetime import datetime
import argparse

NUM_MELS = 32
NUM_FRMS = 40    

'''
function: pad_array

Pads a 2-d array by adding the specified number of low value vectors before, after
above and below
''' 
def pad_array (in_array, pad_before, pad_after, pad_above, pad_below, padding_val):
    len_padded = in_array.shape[1] + pad_before + pad_after
    out_arr = np.ones((in_array.shape[0]+pad_above+pad_below,len_padded))*padding_val
    #print('pad_array: len_unpadded={}, pad_before={}, pad_after={}'.format(in_array.shape[1], pad_before, pad_after))
    if len_padded == len(in_array):
        # nothing to do - just return the input vector
        return in_array
        
    else:
        #out_arr[:, pad_before:(pad_before+in_array.shape[1])] = in_array
        out_arr[pad_below:(pad_below+in_array.shape[0]), pad_before:(pad_before+in_array.shape[1])] = in_array
        return out_arr
        
''' 
function: horiz_xcorr
Compute the spectrogram cross correlation of two flat features with the specified amount of
padding in vertical and horizontal dimensions
'''
def horiz_xcorr(feat1, feat2, horiz_pad, vert_pad):
    # reshape flat feature rows into 32x40 arrays
    clip1 = np.reshape(feat1, (-1,NUM_MELS,NUM_FRMS,1), order='F')[0,:,:,0]
    clip2 = np.reshape(feat2, (-1,NUM_MELS,NUM_FRMS,1), order='F')[0,:,:,0]
    #print(clip1.shape, clip2.shape)
    # pad clip1 out in both horizontal directions by the 'full' amount
    clip1_padded = pad_array(clip1,horiz_pad,horiz_pad,vert_pad,vert_pad,0.0)
    #print(clip1_padded.shape)
    if vert_pad == 0:
        return correlate(clip1, clip2, mode='full')
    else:
        return correlate(clip1_padded, clip2, mode='valid')
    
def main():
    parser = argparse.ArgumentParser(
        description='Compute SPectrogram Cross Correlation (SPCC)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to process spectrogram clips for',
                        type=str)
    parser.add_argument('--freqs', '-f',
                        help='number of frequency bins to shift over',
                        type=int, default=6)
    parser.add_argument('--song-call', '-s',
                        help='does species have distinct song and call (1=yes), 0=no)',
                        type=int, default=1)
    args = parser.parse_args()


    args = parser.parse_args()
    species = args.species
    vert_range = args.freqs
    song_call = args.song_call
    
    src_dir = pathlib.Path('../dataset/audio')
    species_folder = os.path.join(src_dir, species)
    
    # get list of all 32x40 features files in analysis folder and select the latest one
    files_list = glob.glob(os.path.join(species_folder, 'analysis', 'feats_32x40*.csv'))
    if len(files_list) == 0:
        print('\nNo 32x40 features files found for', species,'!!!')
        sys.exit()
        
    feats_file = max(files_list, key=os.path.getmtime)
    try:
        feats = np.loadtxt(feats_file, delimiter=',')
    except Exception as err:
        print('Error opening feats file:', feats_file, err)
        sys.exit()
    print('features file:', feats_file, ':', feats.shape)

    # load src_file and BVP details from the bvp_ids file that match feats_file
    bvp_ids_file = feats_file.replace('feats_32x40','bvp_ids')
    df_ids = pd.read_csv(bvp_ids_file)
    print('BVP IDs file:', bvp_ids_file, ':', df_ids.shape)
    
    if song_call==1 and 'meta_type' not in df_ids.columns:
        # get list of all _bvp files in analysis folder and select the latest one
        bvp_files_list = glob.glob(os.path.join(species_folder, 'analysis', species + '_bvp*.csv')) 
        bvp_file = max(bvp_files_list, key=os.path.getmtime)
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
        
    ### main loop
    tic = time.perf_counter()
    
    dat_size = feats.shape[0]
    cnt = 0
    peak_xcorr = np.zeros((dat_size,dat_size))
    for i in range(dat_size):
        for j in range(i,dat_size):
            cross_correlation = horiz_xcorr(feats[i,:], feats[j,:], NUM_FRMS-1, vert_range)
            peak_xcorr[i,j] = np.max(cross_correlation)
            cnt += 1
            
    # normalize
    norm_xcorr = np.zeros((dat_size,dat_size))
    sum_xcorr = 0
    cnt_xcorr = 0
    for i in range(dat_size):
        for j in range(i+1,dat_size):
            norm_xcorr[i,j] = peak_xcorr[i,j]/np.sqrt(peak_xcorr[i,i]*peak_xcorr[j,j])
            sum_xcorr += norm_xcorr[i,j]
            cnt_xcorr += 1
            
    toc = time.perf_counter()
    print(f"{cnt} normalized peak xcorrs took {(toc - tic):0.4f} seconds")
    avg_xcorr = sum_xcorr / cnt_xcorr
    print(f"average norm_xcorr = {avg_xcorr:0.4f}")
    print(f"complexity score = {(1-avg_xcorr):0.4f}")
    
    if song_call==1:
        '''
        Extract norm_xcorr values for song and call labeled clips and compute a 
        song/call weighted average xcor value
        '''
        df_calls = df_ids[df_ids.meta_type == 'call']
        # extract normalized peak xcorr array for just the call labeled clips
        idx = 0
        num_calls = len(df_calls.index)
        call_xcorr = np.zeros((num_calls,num_calls))
        call_idx_list = df_calls.index.tolist()
        sum_xcorr = 0
        cnt_xcorr = 0
        for i in range(num_calls):
             for j in range(i+1,num_calls):
                    call_xcorr[i,j] = norm_xcorr[call_idx_list[i],call_idx_list[j]]
                    sum_xcorr += norm_xcorr[call_idx_list[i],call_idx_list[j]]
                    cnt_xcorr += 1
                    
        call_avg = sum_xcorr / cnt_xcorr
        print('average peak normalized call_xcorr =', call_avg)        
        
        df_songs = df_ids[df_ids.meta_type == 'song']
        # extract normalized peak xcorr array for just the song labeled clips
        idx = 0
        num_songs = len(df_songs.index)
        song_xcorr = np.zeros((num_songs,num_songs))
        song_idx_list = df_songs.index.tolist()
        sum_xcorr = 0
        cnt_xcorr = 0
        for i in range(num_songs):
             for j in range(i+1,num_songs):
                    song_xcorr[i,j] = norm_xcorr[song_idx_list[i],song_idx_list[j]]
                    sum_xcorr += norm_xcorr[song_idx_list[i],song_idx_list[j]]
                    cnt_xcorr += 1
                    
        song_avg = sum_xcorr / cnt_xcorr      
        print('average peak normalized song_xcorr =', song_avg)
        
        # compute label weighted xcorr
        num_calls = len(df_calls.index)
        num_songs = len(df_songs.index)
        weighted_xcorr = call_avg*num_calls/(num_songs+num_calls) + song_avg*num_songs/(num_songs+num_calls)

        print(f"song/call label weighted_xcorr = {weighted_xcorr:0.4f}")
        print(f"weighted complexity score = {(1-weighted_xcorr):0.4f}")
        
    # save xcorr results for species
    timestamp = datetime.now().strftime("%Y%m%d-%H%M-%S")
    np.savetxt(os.path.join(species_folder, 'analysis', 'xcorr_'+timestamp+'.csv'), norm_xcorr, delimiter=",")    
    
main()    