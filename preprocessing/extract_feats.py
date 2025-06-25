"""
extract_feats.py

This file contains a Python script which is executed per species to extract fixed length 
features of specified resolution that are used for downstream processing (e.g., dimensionality 
reduction and clustering). It is evolved from cluster_feat.py and hi_res_feat.py
03-30-2024: Renaming to extract_feats.py to better reflect functionality in prep for public exposure

The updates in this version are the following:
* takes the frequency resolution of the spectrogram as an input argument (number of Mel passbands)
* time resolution of a spectrogram is the FFT_SIZE/SR. SR is to remain fixed at 22,050 Hz
* the remaining free parameter is the duration of the feature which will remain fixed at 0.9288s in this version

06-24-2025
- minor changes to move execution path down one level as part of folder restructuring for phase 2

10-12-2024
- fixed bug where hard-coded highpass filter coefficents for 1kHz was still being used on the audio
before spectrogram generation

10-10-2024
- added commandline parameter for the MIN_FREQ to be used in spectrogram generation
- updated format of commandline parameters

03-26-2024:
- Revisiting this utility in search of a bug. Noticed features while assessing vade and vae 
based outlier detection where the feature does _not_ correspond to the audio (see backlog item 
#20240325-1 (resolved in extract_bvp.py)
- also cleaning up (e.g., removing unused function defs) and documenting this code in the process
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
from scipy.signal import butter, lfilter, buttord, freqz
import librosa
import argparse

import warnings
warnings.filterwarnings('ignore')

# function to pad a vector (1-D array) by adding the specified low values to beginning and end 
#     and keeping the input vector components in the center locations
def pad_1d (in_vec, padding_before, padding_after, padding_val):
    len_unpadded = len(in_vec)
    out_vec = np.ones(len_unpadded+padding_before+padding_after)*padding_val
    
    out_vec[padding_before:(padding_before+len_unpadded)] = in_vec

    return out_vec
    
# function to pad a 2-d array by adding the specified number of low value vectors before and after 
def pad_array (in_array, pad_before, pad_after, padding_val):
    len_padded = in_array.shape[1] + pad_before + pad_after
    out_arr = np.ones((in_array.shape[0],len_padded))*padding_val
    #print('pad_array: len_unpadded={}, pad_before={}, pad_after={}'.format(in_array.shape[1], pad_before, pad_after))
    if len_padded == len(in_array):
        # nothing to do - just return the input vector
        return in_array
        
    else:
        #out_arr[:, pad_before:(pad_before+in_array.shape[1])] = in_array
        out_arr[:, pad_before:(pad_before+in_array.shape[1])] = in_array
        return out_arr
    
# function to pad a vector (1-D array) to the specified length by adding low values to either end 
#     and keeping the input vector components in the center locations
def pad_vector (in_vec, len_padded, padding_val):
    out_vec = np.ones(len_padded)*padding_val
    
    len_unpadded = len(in_vec)
    padding = len_padded - len_unpadded
    if padding > 0:
        padding_after = int(np.ceil(padding/2.0))
        padding_before = int(np.floor(padding/2.0))
        #print(padding_before, padding_after)
        out_vec[padding_before:(padding_before+len_unpadded)] = in_vec
        
        return out_vec, padding_before, padding_after
    
    else: # nothing to do - just return the input vector
        return in_vec, 0, 0
        
# function returns the number of frames to span input duration
def num_frames(start, dur, sr, frm_size):
    start_frm = math.floor(start*sr/frm_size)
    end_frm = math.floor(start_frm + dur/frm_size)
    
    return math.ceil(dur*sr/frm_size)

# function returns the center of mass (energy) of the input 1D array as offset from start_idx
def center_of_energy(ste, start_idx, end_idx):
    coe = np.dot(ste[start_idx:end_idx], np.arange(start_idx, end_idx)) / np.sum(ste[start_idx:end_idx]) - start_idx
    return coe

"""
function: feat_search
This function searches the input BVP for a fixed duration feature that 
a) has a center of energy (COE) in the center of the feature duration 
b) has the highest total energy of all the features satisfying criterion (a)

Inputs:
* ste - Short-term energy vector
* feat_dur - feature duration

Outputs:
* start_idx
* end_idx

Notes:
* handles case where len(ste) < feat_len
* 03-26-2024: rotates a padded version of ste vector and masks it until the coe is at the center
of the feature length and the total energy is maximized
"""
def feat_search(ste, feat_len):
    ste_padded = pad_vector(ste, len(ste)+feat_len, 0)[0]
    mask = np.concatenate((np.ones(feat_len), np.zeros(len(ste))))
    max_e = 0.0
    ste_rot = ste_padded
    idx_off = int(feat_len/2)
    start_idx = -32768 # initialize to an outragious index value to catch "nothing found" case
    end_idx = -32768 # initialize to an outragious index value to catch "nothing found" case
    
    for i in range(0, len(ste)):
        feat = np.multiply(mask,ste_rot)
        #print(center_of_energy(feat,0,feat_len), feat)
        # if the coe is within tolerance of the center of feature
        if np.abs(center_of_energy(feat,0,feat_len) - (feat_len/2)) < 0.05*feat_len:
            e = np.sum(feat)
            if e > max_e:
                max_e = e
                start_idx = i - idx_off
                #end_idx = min(start_idx + feat_len, start_idx + len(ste))
                end_idx = start_idx + feat_len
                #print('feat_search: max_e update: coe={}, i={}, max_e={:.6f}'.format(center_of_energy(feat,0,feat_len),i,max_e) )
            
        ste_rot = np.roll(ste_rot,-1)
        
    return start_idx, end_idx

"""
Main loop

The following code loops thru all BVPs in a species _bvp file and processes as follows:

    Screen BVPs to only consider those with

            SINR > min_sinr
            SINR > (75th_percentile(BVPs in file) - foreground_thres)

* If the BVP duration is greater than or equal to the fixed feature duration, extract a 
fixed length section of the BVP that has a) center of energy (COE) in the center of the 
feature and b) the highest total energy of the sections that meet criterion (a)
* If the BVP duration is less than the fixed feature duration, pad it to the fixed 
feature duration such that the COE falls in the center

"""
def main():
    parser = argparse.ArgumentParser(
        description='Extract spectrograms from screened segments of one-species bird sounds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to process BVPs for',
                        type=str)
    parser.add_argument('--sinr',
                        help='minimum SINR (dB)',
                        type=float, default=10)
    parser.add_argument('--fg-thresh',
                        help='foreground threshold',
                        type=float, default=5)
    parser.add_argument('--n-frms', '-f',
                        help='number of time frames',
                        type=int, default=40)
    parser.add_argument('--n-mels', '-m',
                        help='number of mels (freq bins)',
                        type=int, default=32)
    parser.add_argument('--min-freq',
                        help='minimum frequency',
                        type=int, default=1000)
    args = parser.parse_args()
    
    species = args.species
    min_sinr = args.sinr
    foreground_thres = args.fg_thresh
    num_frms = args.n_frms
    num_mels = args.n_mels
    min_freq = args.min_freq
    
    print('\n ************************** Run Arguments ****************************')
    print('bird species:',species,'\n min SINR=',min_sinr,'\n foreground_thres=',foreground_thres)
    print(' number of frames=',num_frms,'\n number of Mel bins=',num_mels,'\n min frequency=',min_freq)
    print(' *********************************************************************')

    base_path = '../dataset/audio'
    src_folder = os.path.join(base_path, species)

    #### Setup parameters and load the <species_name>_bvp.csv file
    # get list of all _bvp files in analysis folder and select the latest one
    bvp_files_list = glob.glob(os.path.join(src_folder, 'analysis', species + '_bvp*.csv')) 
    bvp_file = max(bvp_files_list, key=os.path.getmtime)
    if bvp_file == '':
        print('No _bvp*.csv file found')
        sys.exit()
        
    # load bvp file
    try:
        df_bvp = pd.read_csv(bvp_file)
        print('processing ', bvp_file)
    except Exception as err:
        print('File read err for ', bvp_file, ':', err)
        sys.exit()

        
    # global constants (all-caps indicate that value needs to remain the same across scripts)
    SR = 22050
    FEAT_DUR = 0.9288 # feature duration in seconds

    '''
    # high pass filter coefficients
    hp_flt_a=[1.,-1.60155115,0.66868408]
    hp_flt_b=[0.81755881,-1.63511762,0.81755881]
    '''
    # setup highpass filter
    nyq = 0.5 * 22050 # nyquist freq
    wp = min_freq/nyq     # passband edge freq
    ws = (min_freq/2)/nyq # stopband edge freq
    N, Wn = buttord(wp, ws, 3, 12)
    hp_flt_b, hp_flt_a = butter(N, Wn, 'high')
    
    fft_size = int((FEAT_DUR/num_frms) * SR)
    
    # Read local_birds_meta.csv for per audio file average SINRs 
    df_meta = pd.read_csv('local_birds_meta.csv')
    df_meta.set_index('ID',inplace=True)
    
    # determine number of bvps that satisfy inclusion criteria (min_sinr)
    df_bvp_in = df_bvp[(df_bvp.sinr > min_sinr)]
    print('_bvp file dataframe shape after SINR screen:', df_bvp_in.shape)
    bvp_id = []     # list to hold src_file and segment_num
    cluster_feats = np.zeros((df_bvp_in.shape[0],num_mels * num_frms)) # Array to hold features
    
    included_bvp_cnt = 0
    pad_val = 0.00000001
    current_file = ''
    file_sinrs = np.zeros(df_bvp.num_segments.max())

    for row in df_bvp_in.itertuples():
        if row.src_file != current_file:
            # get the bvps of the next src_file 
            current_file = row.src_file
            df_file = df_bvp_in[df_bvp_in.src_file==current_file]
            file_sinrs[:df_file.shape[0]] = df_file.sinr.to_numpy()
            file_sinr_top = np.quantile(file_sinrs[:df_file.shape[0]], 0.75)
            
            # load the source file and apply highpass filter
            file_path = os.path.join(src_folder, current_file)
            try:
                x_raw, sr = librosa.load(file_path)
            except Exception as err:
                print('File read err for ', file_path, ':', err)
                continue

            # pass waveform thru highpass filter
            x = lfilter(hp_flt_b, hp_flt_a, x_raw)
        
        # check SINR of BVP is above the minimum (foreground threshold) below 75 percentile SINR of all BVPs in file 
        if row.sinr > (file_sinr_top - foreground_thres):
            print(row.src_file, row.segment_num, f'duration={row.duration:0.2f}', f'sinr={ row.sinr:0.2f}', f'file top sinr={file_sinr_top:0.2f}')

            dur_frms = num_frames(row.time_offset, row.duration, SR, fft_size)

            # get index into x (i.e., sample indices) and frame indices 
            bvp_start_smp = math.floor(row.time_offset*sr)
            bvp_end_smp = math.ceil((row.time_offset+row.duration)*sr)
            bvp_start_frm = math.floor(bvp_start_smp/fft_size)
            bvp_end_frm = math.ceil(bvp_end_smp/fft_size)
            ste_vec = librosa.feature.rms(y=x[bvp_start_smp:bvp_end_smp], 
                                          frame_length=fft_size, 
                                          hop_length=fft_size, 
                                          center=False)
            S = librosa.feature.melspectrogram(y=x[bvp_start_smp:bvp_end_smp],
                                            sr=sr,
                                            n_fft=fft_size,
                                            hop_length=fft_size, 
                                            n_mels=num_mels, 
                                            htk=True, 
                                            fmin=min_freq,
                                            center=False,
                                            fmax=SR/2) 
            STE = np.transpose(ste_vec)[:,0]
            feat_start, feat_end = feat_search (STE, num_frms)
            if feat_start == -32768: # check for outragious index value to catch "nothing found" case
                # just skip this BVP
                print('feature not found - skipping')
                continue
            
            #print('feat_start={}, feat_end={}'.format(feat_start,feat_end))

            pad_before = 0
            pad_after = 0
            if dur_frms > num_frms:
                # this is the normal case

                # COE at beginning of BVP so feature needs padding at start
                if feat_start < 0:
                    pad_before = -1*feat_start
                    # adjust STE
                    STE = pad_1d(STE[:feat_end],pad_before,0,pad_val)
                    S = pad_array(S[:,:feat_end],pad_before,0,pad_val)
                    #print('BVP duration > feat_len: padding before={}'.format(pad_before))

                # COE at end of BVP so feature needs padding at end
                elif feat_end-feat_start > len(STE[feat_start:]):
                    pad_after = (feat_end-feat_start) - len(STE[feat_start:])
                    # adjust STE & S
                    STE = pad_1d(STE[feat_start:],0,pad_after,pad_val)
                    S = pad_array(S[:,feat_start:],0,pad_after,pad_val)
                    #print('BVP duration > feat_len: padding after={}'.format(pad_after))
                else:
                    # adjust STE & S
                    STE = STE[feat_start:feat_end]
                    S = S[:,feat_start:feat_end]
                    #print('BVP duration > feat_len: no padding')

            elif dur_frms < num_frms:
                # this is the classic padding case (BVP too short so feature _may_ need padding both before and after)
                if feat_start < 0:
                    pad_before = -1*feat_start
                    feat_start = 0

                if feat_end > len(STE):
                    pad_after = feat_end - len(STE)
                    feat_end = len(STE)

                # pad STE
                STE = pad_1d(STE[feat_start:feat_end], pad_before, pad_after, pad_val)
                # pad S
                S = pad_array(S[:,feat_start:feat_end], pad_before,pad_after, pad_val)
                feat_start = 0
                feat_end = num_frms
                short = 1
                #print('BVP duration < feat_len: padding before={} after={}'.format(pad_before, pad_after))


            flat_feat = S.flatten(order='F')
            # Scale feature vector to unity peak magnitude
            cluster_feats[included_bvp_cnt,:len(flat_feat)] = flat_feat/np.linalg.norm(flat_feat,ord=np.inf)
            bvp_id.append([row.src_file, row.segment_num, feat_start*fft_size/SR, feat_end*fft_size/SR])
            included_bvp_cnt += 1

    print(cluster_feats[:included_bvp_cnt,:].shape, 'included', included_bvp_cnt)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M-%S")
    
    # write src_file and segment_num data (IDs) to a .csv
    ids_file = os.path.join(src_folder, 'analysis', 'bvp_ids_'+timestamp+'.csv')
    with open(ids_file, 'w', newline='') as f: 
          
        # using csv.writer method from CSV package 
        write = csv.writer(f) 
          
        write.writerow(['src_file','segment_num','cluster_feat_start','cluster_feat_end','coe_idx','pad_before','pad_after']) 
        write.writerows(bvp_id) 

    # spectrogram resolution string
    res = str(num_mels)+'x'+str(num_frms)
    cluster_feat_file = os.path.join(src_folder, 'analysis', 'feats_'+res+'_'+timestamp+'.csv')
    np.savetxt(cluster_feat_file, cluster_feats[:included_bvp_cnt,:], delimiter=",")        
    
main()