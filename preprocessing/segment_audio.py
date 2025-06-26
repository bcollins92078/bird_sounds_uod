"""
segment_audio.py
Python script automates the extraction of Bird Vocalization Phrases (BVPs) from each audio recording for a 
specified bird species.
03-30-2024: Renaming to segment_audio.py (from extract_bvp.py) to better reflect functionality 
in prep for public exposure

Inputs:
* bird species (common name used for folder names)

Outputs:
* <bird_species>_bvp.csv file containing the segmented audio with the following columns
    + src_file	
    + segment_num	
    + num_segments	
    + time_offset	
    + duration	
    + sinr	

06-24-2025
- minor changes to move execution path down one level as part of folder restructuring for phase 2

10-10-2024
* updated commandline argument format (argparse)
* added commandline argument for minimum frequency
    
03-27-2024:
* fixed bug #20240325-1 (search for note with that number in title)
* removed all frame based metrics from the file, namely
    + energy vector
    + mel spectrogram vectors 
This metrics were no longer being used (moved on to higher resolution spectrograms)

03-28-2024:
* add meta_type column to _bvp file
"""

import argparse
import sys
from sys import argv
import os
import re
import pathlib
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import itertools
from scipy.signal import butter, lfilter, buttord, freqz
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

"""
segment_avg_ergs()

Function to calculate average energy for each segment of a previously segmented signal 

Arguments:
    ste_db - numpy array w/ normalized short-term energy (ste) per frame in dB relative to peak frame energy
    seg_list - list of pairs of ints representing the start and end indices of each bvp segment

Method:
    1) Compute the total energy of the non-BVP segments of the signal and divide that by the total number of frames
    in the non-BVP segments to get the average energy for the non-BVP portion of the waveform. 
    2) Do the same thing for each of the BVP segments to get per-BVP average energy.

Return
    list with the average energies starting with the non-BVP average energy in linear units

"""
def segment_avg_ergs(ste_db, seg_list):
    
    non_bvp_sum_e = 0
    non_bvp_frms = 0
    bvp_avg_list = []
    idx = 0
    prev_end_idx = 0

    for bvp_start_idx,bvp_end_idx in seg_list:
        bvp_avg_erg = np.mean(10**(ste_db[bvp_start_idx:bvp_end_idx+1]/10))
        bvp_avg_list.append(bvp_avg_erg)
        non_bvp_sum_e += np.sum(10**(ste_db[prev_end_idx:bvp_start_idx]/10))
        non_bvp_frms += bvp_start_idx - prev_end_idx
        #print('segment_avg_ergs: idx =',idx,'[bvp_start_idx bvp_end_idx]=[',bvp_start_idx,bvp_end_idx,'], bvp_avg_erg=',bvp_avg_erg,'non_bvp_sum_e=',non_bvp_sum_e)
        idx += 1
        prev_end_idx = bvp_end_idx
        
    # account for non-bvp after the final bvp
    non_bvp_sum_e += np.sum(10**(ste_db[prev_end_idx:]))
    non_bvp_frms += len(ste_db) - prev_end_idx
    
    return bvp_avg_list, non_bvp_sum_e/non_bvp_frms
    
"""
detect_silence()
Function to segment based on silence

This function duplicates pydub.silence.detect_silence and pydub.silence.detect_nonsilent 
without using audio_segment data type. Input the time series as an Numpy array instead.

Arguments:
* ste_vec - the short-term energy series to find silence in
* min_silence_len - the minimum length for any silent section
* silence_thresh - the upper bound for how quiet is silent in dB
* seek_step - step size for interating over the segment in frames

Outputs:
* Returns a list of all silent sections [start, end] in seconds of audio.
"""
def detect_silence(ste_vec, min_silence_len=20, silence_thresh=-16, seek_step=1):
    seg_len = len(ste_vec)

    #print('detect_silence: inputs', seg_len, min_silence_len, silence_thresh, seek_step)
    
    # you can't have a silent portion of a sound that is longer than the sound
    if seg_len < min_silence_len:
        #print('detect_silence: seg_len =', seg_len, '< min_silence_len =', min_silence_len)
        return []

    # find silence and add start and end indicies to the to_cut list
    silence_starts = []

    # check successive (1 frame by default) chunk of sound for silence
    # try a chunk at every seek_step
    last_slice_start = seg_len - min_silence_len
    slice_starts = range(0, last_slice_start + 1, seek_step)

    # guarantee last_slice_start is included in the range
    # to make sure the last portion of the audio_ts is searched
    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])

    for i in slice_starts:
        ste_slice = ste_vec[i:i + min_silence_len]
        if np.max(ste_slice) <= silence_thresh:
            silence_starts.append(i)

    # short circuit when there is no silence
    if not silence_starts:
        #print('detect_silence: no silence_starts')
        return []

    # combine the silence we detected into ranges (start - end)
    silent_ranges = []

    prev_i = silence_starts.pop(0)
    current_range_start = prev_i

    for silence_start_i in silence_starts:
        continuous = (silence_start_i == prev_i + seek_step)

        # sometimes two small blips are enough for one particular slice to be
        # non-silent, despite the silence all running together. Just combine
        # the two overlapping silent ranges.
        silence_has_gap = silence_start_i > (prev_i + min_silence_len)

        if not continuous and silence_has_gap:
            silent_ranges.append([current_range_start,
                                  prev_i + min_silence_len])
            current_range_start = silence_start_i
        prev_i = silence_start_i

    silent_ranges.append([current_range_start,
                          prev_i + min_silence_len])

    #print('detect_silence: output', silent_ranges)
    return silent_ranges

"""
silent_to_nonsilent
This function converts silence segments to non-silence segments

Inputs:
* silent_ranges - list of silent ranges
* ste_db - short-term-energy vector

Outputs:
* nonsilent_ranges - list of non-silent ranges
"""
def silent_to_nonsilent(silent_ranges, ste_db):
    # convert silent_segs to nonsilent_segs
    len_seg = len(ste_db)

    # if there is no silence, the whole thing is nonsilent
    if not silent_ranges:
        return [[0, len_seg]]

    # short circuit when the whole audio segment is silent
    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
        return []

    prev_end_i = 0
    nonsilent_ranges = []
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append([prev_end_i, start_i])
        prev_end_i = end_i

    if end_i != len_seg:
        nonsilent_ranges.append([prev_end_i, len_seg])

    if nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)

    return nonsilent_ranges
    
"""
segment_diy

This function uses the pyDub.silence.detect_silence clone function to segment based on silences. 
Silence threshold is an offset from an estimate of the noise floor of the recording.

Inputs:
* ste_db - the normalized (peak = 0 dB) short-term-energy vector in dB units 
* min_silence_len - the minimum number of frames that must be 
below the silence threshold to qualify as a silent interval (default: 11 = 0.5s)
* mult_offset - multiplier used to compute silence_threshold from the noise floor estimate

Outputs:
* list of non-silent segments in [start_idx, end_idx] format
* noise floor estimate
"""
def segment_diy(ste_db, min_silence_len=11, mult_offset=0.66):
    # estimate the noise_floor as the 25th percentile of the input data series
    noise_floor = np.quantile(ste_db, 0.25)
    silent_segs = detect_silence(ste_db, min_silence_len=min_silence_len, silence_thresh=noise_floor*mult_offset)
    
    # convert silent_segs to nonsilent_segs
    nonsilent_segs = silent_to_nonsilent(silent_segs, ste_db)
    #print('segment_diy: noise_floor={:.2f} num_segs={}'.format(noise_floor, len(nonsilent_segs)), nonsilent_segs)
    return noise_floor, nonsilent_segs


"""
extract_file_bvps()
Function to extract BVPs from audio file. Function to load audio time series, segment it 
and extracts features from each segment
Arguments:
    file_path - path to audio file to be processed
    df_bvp   - dataframe to return results
    FRM_SIZE - frame size (default: 1024)
    MIN_FREQ - minimum frequency (default: 1000)
    NUM_MELS - number of Mel passbands (default: 16)
    HOP_SIZE - number of samples to "hop" between frames (determines frame overlap. default: 1024)
    N_FFT - FFT size in computing spectrogram (default: 1024)
Method:
    apply highpass filter to audio sequence read from specified file (file_path)
    cloned pyDub.silence.detect_silence() to segment audio short-term energy (STE) sequence
    librosa.feature.rms to compute short-term energy
    librosa.feature.melspectrogram to compute mel-spectrogram
    segment_avg_ergs to compute basis for BVP SINR (revised version of segment_avg_energies)
Return
    pandas dataframe with the BVPs extracted
    segments (list of lists of frame index pairs indicating start and end of each segment)
    spectrogram
"""
def extract_file_bvps(file_path, df_bvp, min_silence_len=11, 
                      FRM_SIZE=1024, min_freq=1000, NUM_MELS=16, HOP_SIZE=1024, N_FFT=1024, 
                      hp_flt_a=[1.,-1.60155115,0.66868408], hp_flt_b=[0.81755881,-1.63511762,0.81755881]):
    try:
        x_raw, sr = librosa.load(file_path)
    except Exception as err:
        print('File read err for ', file_path, ':', err)
        return

    # pass waveform thru highpass filter
    x = lfilter(hp_flt_b, hp_flt_a, x_raw)
            
    file = os.path.basename(file_path)
    
    # call librosa.feature.rms() to get short-term energies squareroot for the recording
    ste_vec = librosa.feature.rms(y=x, frame_length=FRM_SIZE, hop_length=HOP_SIZE, center=False)
    # normalize the mean of squares value instead of root mean square computed above and convert to dB
    ste_db = np.clip(10*np.log10(((ste_vec/np.max(ste_vec))[0])**2), -80, 0)
    
    # call segment_diy() to get list of segments 
    # Note: this function returns segments as pairs of indices instead of time values
    noise_floor, segments = segment_diy(ste_db, min_silence_len=11)
    
    #print('extract_file_bvps found', len(segments), 'segments in', file_path, ':', segments)
    
    # compute segment energies
    seg_energies, out_energy = segment_avg_ergs(ste_db, segments)

    # build df_bvp rows
    seg_cnt = 0
    for seg in segments:
        # calculate index range for BVP
        bvp_start_frm = seg[0]
        bvp_end_frm = seg[1]

        seg_cnt += 1
        row = []
        row.append(file)          # src_file
        row.append(seg_cnt)       # segment_num
        num_segs = len(segments)
        row.append(num_segs)      # num_segments
        row.append(seg[0]*FRM_SIZE/sr)        # time_offset
        row.append((seg[1]-seg[0])*FRM_SIZE/sr) # duration

        row.append(10*np.log10(seg_energies[seg_cnt-1]/out_energy)) # sinr

        # add flag for segmentations to inspect
        if num_segs < 2:
            row.append('one segment')
            print('extract_file_bvps: flagged {} one segment, noise_floor = {:.2f}'.format(file, noise_floor))
        elif float(num_segs/len(ste_vec[0])) > 0.333:    # more than 1 segment for every 3s
            row.append('many segments')
            print('extract_file_bvps: flagged {} many segments({}), noise_floor = {:.2f}'.format(file, num_segs, noise_floor))
        elif float((seg[1]-seg[0])/len(ste_vec[0])) > 0.5: # segment too long
            row.append('long segment')
            print('extract_file_bvps: flagged {} long segment({}), noise_floor = {:.2f}'.format(file, seg_cnt, noise_floor))
        else:
            row.append('')
            
        row.append('')  # meta_type
            
        # add the row to new dataframe
        a_series = pd.Series(row, index = df_bvp.columns)
        df_bvp = df_bvp.append(a_series, ignore_index = True)

        #print(file, seg_cnt, S.shape, seg[1]-seg[0])
    return segments, df_bvp
    
"""
main()
Function segments each audio file in the ./dataset/audio/<bird_species>/ directory to remove silence segments 
and extracts features from each segment. Results are output to a <bird_species>_bvp.csv file located in the 
../dataset/audio/<bird_species>.analysis folder
"""
def main():
    parser = argparse.ArgumentParser(
        description='Segment audio from one-species bird sound recordings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to process BVPs for',
                        type=str)
    parser.add_argument('--min-silence-len',
                        help='minimum silence between segments (time frames)',
                        type=int, default=11)
    parser.add_argument('--min-freq',
                        help='minimum frequency',
                        type=int, default=1000)
    args = parser.parse_args()
    
    bird_species = args.species
    min_silence_len = args.min_silence_len
    min_freq = args.min_freq
    
    print('\n ************************** Run Arguments ****************************')
    print('bird species:',bird_species,'\n min silence=',min_silence_len)
    print(' min frequency=',min_freq)
    print(' *********************************************************************')

    tic = time.perf_counter()
    base_path = '../dataset/audio'

    # Does the bird species parameter point to an appropriate directory struct
    species_folder = os.path.join(base_path, bird_species)
    if os.path.isdir(species_folder) == False:
        print('Error: {%s} doesn\'t exist', species_folder)
        sys.exit()
        
    # Does the analysis subdirectory exist?
    anal_path = os.path.join(species_folder, 'analysis')
    if os.path.isdir(anal_path) == False:
        print('Analysis folder doesn\'t exist. Creating', anal_path)
        try:
            os.mkdir(anal_path)
        except Exception as err:
            print('Error creating analysis subdir:', anal_path, err)
            sys.exit()
            
    # Read local_birds_meta.csv for per audio file meta_type 
    df_meta = pd.read_csv('local_birds_meta.csv')
    # convert collapsed species common name back to name with spaces used in meta_file
    str_list= re.findall('[a-z][A-Z]',bird_species)
    cmn_name = bird_species
    for s in str_list:
        cmn_name = cmn_name.replace(s, s[0]+' '+s[1])
    # filter meta_file to contain only the records for this species
    df_meta = df_meta[df_meta.Common_Name==cmn_name]
    df_meta.set_index('ID',inplace=True)
    
    #### Setup parameters 
    #MIN_FREQ = 1000
    NUM_MELS = 16
    SR = 22050
    FRM_SIZE = 1024
    N_FFT = 1024        
    HOP_SIZE = 1024       
    WIN_SIZE = 1024      
    WINDOW_TYPE = 'hann'
    
    # setup highpass filter
    nyq = 0.5 * 22050 # nyquist freq
    wp = min_freq/nyq     # passband edge freq
    ws = (min_freq/2)/nyq # stopband edge freq
    N, Wn = buttord(wp, ws, 3, 12)
    hp_flt_b, hp_flt_a = butter(N, Wn, 'high')
    
    df_bvp = pd.DataFrame(columns=['src_file', 'segment_num', 'num_segments','time_offset', 'duration',
                           'sinr', 'inspect', 'meta_type'])
    
    species_folder = os.path.join(base_path, bird_species)
    src_files = (file for file in os.listdir(species_folder) if os.path.isfile(os.path.join(species_folder, file)))
   
    bvp_cnt = 0
    for file in src_files:
        if file[-4:] == '.mp3':
            file_path = os.path.join(species_folder, file)
            #print(file)
            segments, df_bvp = extract_file_bvps(file_path, df_bvp, min_silence_len=min_silence_len, min_freq=min_freq)
            try:
                df_bvp.loc[df_bvp[df_bvp.src_file==file].index,'meta_type'] = df_meta.at[int(file.replace('.mp3', '')), 'Rec_Content']
            except Exception as err:
                print('Error processing file', file, ':', err)
                continue
            bvp_cnt += len(segments)

    # write df_bvp for the species to the species analysis folder
    date_str = datetime.now().strftime("%Y%m%d")
    bvp_file = os.path.join(anal_path, bird_species+'_bvp'+'_'+date_str+'.csv')
    df_bvp.to_csv(bvp_file, index=False)
    
    bvp_dur = df_bvp.duration.sum()
    print(bird_species, ': Total BVP count={:d}, total duration={:0.2f}'.format(bvp_cnt, bvp_dur))
    
    toc = time.perf_counter()
    print(f"Extracting BVPs for {bird_species} species took {(toc - tic)/60:0.4f} minutes")
    
main()