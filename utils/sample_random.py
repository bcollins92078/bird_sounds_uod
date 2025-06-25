"""
sample_random.py

Python script randomly samples the specified species' dataset and presents each
sample to the user for a decision - inlier, outlier or indeterminant.
i. randomly selects num_samples from the species's screened dataset population for 95% CI and 5% error
ii. Plot the segment that the sample is from (and highlight the section that is the sample - TODO)
iii. Play the audio of the entire segment 
iv. Prompt the user for inlier, outlier or undetermined along with an optional comment
and record the answer in an "manual" column and the comment both added to the df_ids table 
v. Repeat steps ii thru iv until the num_samples plus the number of undetermined decisions has been reached
vi. Calculate the outlier rate and record it 
vii. Output the df_ids table and the computed summary stats 

06-25-2025
- Found this file in text editor so assuming that it is incomplete! Added minor change to move
execution path down one level as part of folder restructuring for phase 2.
"""

import os
import sys
import pathlib
import glob
import pandas as pd
import numpy as np
import math
import time
import argparse
import simpleaudio as sa
import re
import librosa
from scipy.signal import butter, lfilter, buttord, freqz
import matplotlib.pyplot as plt
from random import seed
from random import random


def main():
    parser = argparse.ArgumentParser(
        description='Sample audio segments flagged as inliers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to sample inliers for',
                        type=str)
    parser.add_argument('filename',
                        help='name of .csv file containing screened samples',
                        type=str)
    parser.add_argument('--proportion', '-p',
                        help='inlier proportion assumed for sample size calculation',
                        type=float, default=0.1)
    parser.add_argument('--out',
                        help='output filename for summary',
                        type=str, default='sample_random.out')
    parser.add_argument('--min-freq',
                        help='minimum frequency',
                        type=int, default=1000)
    args = parser.parse_args()
    species = args.species
    filename = args.filename
    p_assumed = args.proportion
    out_file = args.out
    min_freq = args.min_freq

    NUM_MELS = 32
    NUM_FRMS = 40

    # setup highpass filter
    nyq = 0.5 * 22050 # nyquist freq
    wp = min_freq/nyq     # passband edge freq
    ws = (min_freq/2)/nyq # stopband edge freq
    N, Wn = buttord(wp, ws, 3, 12)
    hp_flt_b, hp_flt_a = butter(N, Wn, 'high')

    # load latest _bvp file
    src_dir = pathlib.Path('../dataset/audio')
    species_folder = os.path.join(src_dir, species)

    bvp_files_list = glob.glob(os.path.join(species_folder, 'analysis', species + '_bvp*.csv')) 
    bvp_file = max(bvp_files_list, key=os.path.getmtime)
    if bvp_file == '':
        print('No _bvp*.csv file found')
        sys.exit()
    else: # load bvp file
        try:
            df_bvp = pd.read_csv(bvp_file)
            print('processing ', bvp_file)
        except Exception as err:
            print('File read err for ', bvp_file, ':', err)
            sys.exit()

    # get list of all 32x40 features files in analysis folder and select the latest one
    files_list = glob.glob(os.path.join(species_folder, 'analysis', 'feats_32x40*.csv'))
    feats_file = max(files_list, key=os.path.getmtime)
    try:
        feats = np.loadtxt(feats_file, delimiter=',')
    except Exception as err:
        print('Error opening feats file:', feats_file, err)
        sys.exit()

    # get the bvp_ids files in analysis folder that corresponds to the feats_file
    bvp_ids_file = os.path.join(species_folder,'analysis', 'bvp_ids_'+feats_file.split('_')[-1])
    try:
        df_ids = pd.read_csv(bvp_ids_file)
    except Exception as err:
        print('Error opening bvp_ids file:', bvp_ids_file, err)
        sys.exit()

    # load discards file assumed to be in the artifacts folder under the species input
    try:
        df_disc = pd.read_csv(os.path.join(species_folder, 'analysis','artifacts',filename))
        print('processing ', os.path.join(species_folder, 'analysis','artifacts',filename))
    except Exception as err:
        print('File read err for ', os.path.join(species_folder, 'analysis','artifacts',filename), ':', err)
        sys.exit()
    
    if 'inlier' not in df_disc.columns:
        df_disc['inlier'] = np.nan
        
    # open a summary file
    f = open(os.path.join(species_folder, 'analysis', 'artifacts', out_file), 'w')
    
    f.write('************************** sample_inliers.py inputs ****************************')
    
    f.write('\nbird species:'+ species)
    f.write('\nBVP file: '+ bvp_file)
    f.write('\nFeatures file: '+ feats_file)
    f.write('\nBVP IDs file: '+ bvp_ids_file)
    f.write('\nDiscards file: '+ filename)
    f.write('\ninlier flags columns: '+ inlier_cols)
    f.write('\n*************************** sample_inliers.py outputs ****************************')
    
    inlier_cols_list = inlier_cols.split(',')
    
    '''
    Compute limited population sample size (equations from https://www.calculator.net/sample-size-calculator.html?type=1&cl=95&ci=5&pp=50&ps=&x=Calculate)
    - population proportion, p, assumed to be 0.5
    - z score for 95% confidence interval is 1.96
    - margin of error is 5%
    - sample size for unlimited population, m = z^2 * p*(1-p)/err^2 = 384.16 so 385
    - sample size for limited population = m/(1+(m-1)/N) where N is the population size
    
    In our case the population size is the number of inliers designated by each of the model ensembles being sampled
    '''
    # loop thru each of the inlier column names and compute sample size
    z = 1.96    # 95% confidence interval
    e = 0.05    # margin of error
    flat_feat = np.zeros(NUM_MELS * NUM_FRMS)    
    for i in range(len(inlier_cols_list)):
        df_in = df_disc[df_disc[inlier_cols_list[i]]==False]
        m = math.ceil(pow(z,2) * p_assumed*(1-p_assumed)/pow(e,2))
        inlier_class_size = len(df_in.index)
        sample_size = math.ceil(m/(1+(m-1)/inlier_class_size))
        f.write('\n'+inlier_cols_list[i]+': assumed proportion='+ str(p_assumed)+', sample_size='+ str(sample_size)+ ' out of ' + str(inlier_class_size))
    
        # get random selection of size sample_size that have been flagged by ensemble as inliers
        if i == 0:
            df_samples = df_in.sample(sample_size)
            print(inlier_cols_list[i],': inlier_class_size=',inlier_class_size,'sample_size=',len(df_samples.index))
        else:
            df_in = df_in.sample(sample_size)
            df_samples = pd.concat([df_samples,df_in]).drop_duplicates()
            print(inlier_cols_list[i],': inlier_class_size=',inlier_class_size,'sample_size=',len(df_in.index))
    
    print('overall sample count=',len(df_samples.index))
    # sort df_samples
    df_samples = df_samples.sort_values(['src_file','segment_num'])
    
    # loop thru df_samples plotting spectrograms and playing segments for user eval
    eval_cnt = 0
    tot_cnt = 0
    in_cnt = 0
    indet_cnt = 0
    prev_file = ''

    for row in df_samples.itertuples():
        eval_cnt += 1
        print('Sample #',eval_cnt,' of',len(df_samples.index), ':', row.src_file, row.segment_num)
        if pd.isna(df_disc.loc[row.Index,'inlier']):
            # read the audio file
            file_path = os.path.join(species_folder, row.src_file)
            if row.src_file != prev_file:
                x_raw, sr = librosa.load(file_path)
                x = lfilter(hp_flt_b, hp_flt_a, x_raw)
                x *= 32767 / max(abs(x))
                x = x.astype(np.int16)
                prev_file = row.src_file
                
            '''
            plot spectrogram of segment and highlight section extracted as the sample
            '''
            # get index to row in feats that corresponds to src_file and segment_num
            idx = df_ids[(df_ids.src_file == row.src_file) & (df_ids.segment_num == row.segment_num)].index.astype(int)[0]
            
            # scale and reshape features for specified BVP
            flat_feat[:] = feats[idx,:]/np.max(feats[idx,:])
            feat = np.reshape(flat_feat, (-1,NUM_MELS,NUM_FRMS,1), order='F')
            
            # plot 
            plt.figure(1)
            plt.imshow(feat[0][:,:,0], origin='lower')
            #plt.show()
            plt.pause(0.001)
            #input("Press [enter] to continue.")
            
            # play audio of the entire segment
            start = df_bvp[(df_bvp.src_file==row.src_file) & (df_bvp.segment_num==row.segment_num)].time_offset.to_numpy()[0]
            dur = df_bvp[(df_bvp.src_file==row.src_file) & (df_bvp.segment_num==row.segment_num)].duration.to_numpy()[0]
            start_idx = int(math.floor(start*sr))
            end_idx = int(math.floor((start + dur)*sr))
            
            repeat = True
            while repeat:
                # play the segment of audio 
                play_obj = sa.play_buffer(x[start_idx:end_idx],1,2,sr)
            
                # prompt user for input (1 for inlier, 0 for inlier, -1 for indeterminant) and optional comment
                resp_str = input('1 = inlier, 0 = inlier, -1 = indeterminant. Optional text string after comma can be entered as comment\n')
                resp_list = resp_str.split(',')
                if resp_list[0].isnumeric() or resp_list[0] == '-1':
                    resp_1 = int(resp_list[0])
                    df_disc.loc[row.Index,'inlier'] = resp_1
                    tot_cnt += 1
                    if resp_1 == 1:
                        in_cnt += 1
                    elif resp_1 == -1:
                        indet_cnt += 1
                    repeat = False
                else:
                    # just repeat the sample playback
                    continue
                if len(resp_list) >= 2:
                    df_disc.loc[row.Index,'comment'] = resp_list[1]
                    repeat = False
        else: # this sample has already been manually evaluated
            resp_1 = df_disc.loc[row.Index,'inlier']
            tot_cnt += 1
            if resp_1 == 1:
                in_cnt += 1
            elif resp_1 == -1:
                indet_cnt += 1
                
    # compute false negative rate (fnr) for each model or ensemble
    print('Overall FNR: ', 1.0-float(in_cnt/tot_cnt), in_cnt, tot_cnt, indet_cnt)
    for i in range(len(inlier_cols_list)):
        df_in = df_disc[(df_disc[inlier_cols_list[i]]==False) & ((df_disc['inlier']==1) | (df_disc['inlier']==0))]
        in_cnt = int(df_in.inlier.sum())
        tot_cnt = len(df_in.index)
        fnr = float(in_cnt/tot_cnt)
        inlier_class_size = len(df_disc[df_disc[inlier_cols_list[i]]==False].index)
        f.write('\n'+inlier_cols_list[i])
        f.write('\n FNR = '+ str(fnr)+' +/- 5% with 95% confidence')
        f.write('\n sample_size achieved='+ str(tot_cnt))
        f.write('\n false negatives sampled='+ str(in_cnt))
        f.write('\n estimated false negatives='+ '['+str(math.ceil(fnr*0.95*inlier_class_size))+','+str(math.ceil(fnr*1.05*inlier_class_size))+']')

    f.close()
    df_disc.to_csv(os.path.join(species_folder, 'analysis','artifacts',filename), index=False)
    
main()