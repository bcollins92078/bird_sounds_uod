'''
process_meta.py
This Python script extracts metadata fields from the page?.json files downloaded from Xeno-Canto
and adds them to local_brids_meta.csv and local_birds_sum.csv.

05-15-2024:
* create analysis and analysis\artifacts subfolders is they don't exist
'''

import argparse
import sys
import os
import shutil
import pandas as pd
import numpy as np
import json
import datetime
import time
import re
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(
        description='Extracts metadata fields from downloaded Xeno-Canto json and add them to .csv files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to process Xeno-Canto metadata for',
                        type=str)
    parser.add_argument('--meta-file', '-m',
                        help='name of recordings metadata output file',
                        type=str, default='local_birds_meta.csv')
    parser.add_argument('--sum-file', '-s',
                        help='name of summary metadata output file',
                        type=str, default='local_birds_sum.csv')

    args = parser.parse_args()
    species = args.species
    meta_file = args.meta_file
    sum_file = args.sum_file

    '''
    create analysis and analysis\artifacts subfolders if they don't exist
    Note: this is for later pipeline processing and these folders are _not_ used here
    '''
    base_path = 'dataset/audio'
    species_folder = os.path.join(base_path, species)
    # verify that species folder has been setup
    if os.path.isdir(species_folder) == False:
        print('Species folder doesn\'t exist as is required!')
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
            
    # Does the artifacts subdirectory exist?
    art_path = os.path.join(anal_path, 'artifacts')
    if os.path.isdir(art_path) == False:
        print('artifacts folder doesn\'t exist. Creating', art_path)
        try:
            os.mkdir(art_path)
        except Exception as err:
            print('Error creating artifacts subdir:', art_path, err)
            sys.exit()
    
    # Does the models subdirectory exist?
    mod_path = os.path.join(anal_path, 'models')
    if os.path.isdir(mod_path) == False:
        print('models folder doesn\'t exist. Creating', mod_path)
        try:
            os.mkdir(mod_path)
        except Exception as err:
            print('Error creating models subdir:', mod_path, err)
            sys.exit()
    '''
    read sum_file
    '''
    # open output summary file to which new fields are to be appended
    try:
        df_summary = pd.read_csv(sum_file)
    except Exception as err:
        print('File read err for ', sum_file, ':', err)
        sys.exit('Error Exit!')
    
    '''
    extract per recording metadata and append to meta_file
    '''
    # open output metadata file to which new fields are to be appended
    try:
        df_recs = pd.read_csv(meta_file,index_col='ID')
    except Exception as err:
        print('File read err for ', meta_file, ':', err)
        sys.exit('Error Exit!')
    
    xc_meta_path = 'dataset/metadata/' + os.fsdecode(species) + '/page1.json'
    print(xc_meta_path)
    with open(xc_meta_path, 'r') as f:
        data = json.load(f)
    df_new = pd.DataFrame(data['recordings'])

    # handle multiple page scenario
    pages = int(data['numPages']) + 1
    for pg in range(2, pages):
        xc_meta_path = 'dataset/metadata/' + os.fsdecode(species) + '/page' + str(pg) + '.json'
        with open(xc_meta_path, 'r') as f:
            data = json.load(f)
        df_inc = pd.DataFrame(data['recordings'])
        print(xc_meta_path)
        df_new = df_new.append(df_inc, ignore_index=True)

    # keep only the columns of interest
    df_new = df_new.drop(['url','file','file-name','sono','lic','group','sex','stage',
                            'method','osci','animal-seen','temp','regnr','auto','dvc','mic','smp'], axis=1)

    # rename columns
    df_new = df_new.rename(columns={"id": "ID", "gen": "Genus", 'sp':'Species', 'ssp':'Sub_Species', 'en':'Common_Name', 
                       'rec':'Recordist', 'cnt':'Country', 'loc':'Location', 'length':'Length', 
                       'type':'Type', 'q':'Quality', 'time':'Time', 'date':'Date','uploaded':'Date_Uploaded', 
                       'also':'Other_Species', 'rmk':'Remarks', 'bird-seen':'Bird_Seen', 'playback-used':'Playback_Used'})
    df_new.set_index('ID', inplace=True)
    
    # Change format of recording length column to integer number of seconds and lat, lng, alt to floats
    df_new.Latitude = np.NaN
    df_new.Longitude = np.NaN
    df_new.Altitude = np.NaN

    for row in df_new.itertuples():
        try:
            x = time.strptime(row.Length,'%M:%S')
        except:
            try:
                x = time.strptime(row.Length,'%H:%M:%S')
            except:
                print('Length value doesn\'t match format', row.Length )

        df_new.loc[row.Index,'Duration']=datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()

        # Change formats of lat and lng columns to float
        try:
            df_new.loc[row.Index, 'Latitude'] = float(row.lat)
        except:
            #print('Bad Latitude:', row.lat)
            df_new.loc[row.Index, 'Latitude'] = np.NaN
            
        try:
            df_new.loc[row.Index, 'Longitude'] = float(row.lng)
        except:
            #print('Bad Longitude:', row.lng)
            df_new.loc[row.Index, 'Longitude'] = np.NaN
            
        try:
            df_new.loc[row.Index, 'Altitude'] = float(row.alt)
        except:
            #print('Bad Altitude:', row.alt)
            df_new.loc[row.Index, 'Altitude'] = np.NaN
    
    # Add new column - Rec_Content
    df_new['Rec_Content']='other'

    for row in df_new.itertuples():       
        pat_song = re.compile('song', re.IGNORECASE)
        pat_call = re.compile('call', re.IGNORECASE)
        
        if pat_song.search(row.Type) != None:
            result = 1
        else:
            result = 0
            
        if pat_call.search(row.Type) != None:
            result += 2
        
        if result == 1:
            df_new.loc[row.Index, 'Rec_Content'] = 'song'
        elif result == 2:
            df_new.loc[row.Index, 'Rec_Content'] = 'call'
        elif result == 3:
            df_new.loc[row.Index, 'Rec_Content'] = 'both'
        else:
            df_new.loc[row.Index, 'Rec_Content'] = 'other'

    df_recs = df_recs.append(df_new)
    df_recs.to_csv(meta_file)

    '''
    complete update entry for sum_file
    '''
    df_sum = pd.Series(index=['Common_Name', 'num_recordings', 'total_duration', 
                                    'num_song', 'num_call', 'num_both', 'num_other'])
    
    # since common_name should be same for all df_new entries...
    df_sum.Common_Name = df_new.iloc[0,3]
    
    df_sum.num_recordings = int(data['numRecordings'])
    
    df_sum.num_song = len(df_new[df_new.Rec_Content == 'song'].index)
    df_sum.num_call = len(df_new[df_new.Rec_Content == 'call'].index)
    df_sum.num_both = len(df_new[df_new.Rec_Content == 'both'].index)
    df_sum.num_other = len(df_new[df_new.Rec_Content == 'other'].index)
    
    df_sum.total_duration = df_new.Duration.sum()
    #print(df_sum)

    df_summary = df_summary.append(df_sum, ignore_index = True)
    df_summary.to_csv(sum_file, index=False)
main()
