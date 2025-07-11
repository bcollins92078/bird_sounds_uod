"""
cvae_clean.py

Python script automates the per-species data cleaning using a convolutional variational autoencoder (CVAE) 
as follows:
* Dimensionality reduction: CVAE is trained on preprocessed spectrograms clips from bird sound recordings
* Hierarchical agglomerative clustering (HAC) is applied to the CVAE latent space representation 
of the clips to produce n_clusters
* Discard candidates are determined at the cluster level with smallest clusters farthest away from 
one of the biggest clusters recommended for discard first and proceeding to larger clusters until
the specified max discard percentage is reached
* The preceding process steps are repeated a specified number of times and each time the 
resulting model parameters, latent space encodings, cluster assignments and discard decisions are saved


06-24-2025
- minor changes to move execution path down one level as part of folder restructuring for phase 2 and 
clean up header comments

08-18-2023
* changed name from clean_bvps.py to vae_clean.py to be consistent with the alternative VADE mechanism
* added more metrics to summary*.csv to upgrade to level achieve in vade_clean.py

05-17-2024
* added homogeneity score to list of metrics computed 
"""
import sys
from sys import argv

import os
import pathlib
import math
import time
from datetime import datetime
import glob
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.cluster import homogeneity_score

import warnings
warnings.filterwarnings('ignore')
'''
### Define architecture "a0" encoder and decoder networks
class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, num_mels, num_frms):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(num_mels,num_frms, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, padding='same', activation='relu', strides=(2, 2)),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=32, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=16*20*64, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(16, 20, 64)),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, padding='same',)
       ]
    )
    
  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

  # BEC: added this function to "define the forward path" required to get model.summary() to work.
  # Overrides the base class call() function.
  def call(self, input_feats):
    mean, logvar = self.encode(input_feats)
    z = self.reparameterize(mean, logvar)
    reconstructed = self.decode(z, apply_sigmoid=True)
    return reconstructed
'''

### Define architecture "a2" encoder and decoder networks
class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, num_mels, num_frms):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(num_mels, num_frms, 1)),
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=128, input_shape=(4*5*32,), activation='relu'),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=4*5*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(4, 5, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=8, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=2, padding='same'),
       ]
    )

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    #return eps * tf.exp(logvar * .5) + mean
    return eps * tf.exp(logvar) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

  # BEC: added this function to "define the forward path" required to get model.summary() to work.
  # Overrides the base class call() function.
  def call(self, input_feats):
    mean, logvar = self.encode(input_feats)
    z = self.reparameterize(mean, logvar)
    reconstructed = self.decode(z, apply_sigmoid=True)
    return reconstructed

### Define the loss function and the optimizer
optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    
def main():
    import argparse 

    parser = argparse.ArgumentParser('Clean BVPs for one species')

    parser.add_argument('species', type=str, help='bird species to process BVPs for')
    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    parser.add_argument('--num-clusters', '-c', type=int, default=50, help='number of clusters (default: 50)')
    parser.add_argument('--iterations', type=int, default=11, help='number of models to train and evaluate (default: 11')
    parser.add_argument('-e','--epochs', type=int, default=80, help='number of epochs to train each model (default: 80')
    parser.add_argument('--batch-size', type=int, default=32, help='training batch size (default: 32')
    parser.add_argument('--max-discards', type=float, default=0.1, help='maximum fraction of BVPs to discard (default: 0.1)')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('-b', '--num-big', type=int, default=2, help='min number of big clusters (default: 2)')
    parser.add_argument('-s', '--song-call', type=int, default=1, help='does species have distinct song and call (1=yes), 0=no)')

    args = parser.parse_args()
    species = args.species
    latent_dim = args.z_dim
    iterations = args.iterations
    n_clusters = args.num_clusters
    max_disc_fract = args.max_discards
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    num_big = args.num_big
    song_call = args.song_call
    
    print('\n ************************** Run Arguments ****************************\n')
    print('bird species:',species,'\n latent dim=',latent_dim,'\n iterations=',iterations,'\n number of clusters=',n_clusters,'\n max discards fraction=',max_disc_fract)
    print(' num_epochs=',epochs,'\n batch_size=',batch_size,'\n learning rate=',lr,'\n num_big=',num_big,'\n song_call=',song_call)
    
    src_folder = '../dataset/audio/' + species
    MIN_FREQ = 1000
    NUM_MELS = 32
    SR = 22050
    FRM_SIZE = 512
    NUM_FRMS = 40
        
    # get list of all _bvp files in analysis folder and select the latest one
    bvp_files_list = glob.glob(os.path.join(src_folder, 'analysis', species + '_bvp*.csv')) 
    bvp_file = max(bvp_files_list, key=os.path.getmtime)
    
    # load bvp file
    try:
        df_bvp = pd.read_csv(bvp_file)
        print('processing ', bvp_file)
    except Exception as err:
        print('File read err for ', bvp_file, ':', err)
    
    # get list of all cluster_feats files in analysis folder and select the latest one
    cluster_files_list = glob.glob(os.path.join(src_folder, 'analysis', 'feats_32x40_*.csv')) 
    cluster_feat_file = max(cluster_files_list, key=os.path.getmtime)
    print(cluster_feat_file)

    # load cluster_feat file
    cluster_feats = np.loadtxt(cluster_feat_file, delimiter=',', dtype='float32')
    print(cluster_feats.shape)
        
    # get list of all bvp_id files in analysis folder and select the latest one
    bvp_ids_files_list = glob.glob(os.path.join(src_folder, 'analysis', 'bvp_ids*.csv')) 
    bvp_ids_file = max(bvp_ids_files_list, key=os.path.getmtime)
    print(bvp_ids_file)

    # load src_file and BVP details from the bvp_ids file that match cluster_feat.csv
    df_ids = pd.read_csv(bvp_ids_file)
    print(df_ids.shape)

    ### Reshape the data and hold out validation samples
    idx_list = list(range(cluster_feats.shape[0]))
    # train:val ratio of 80:20
    val_size = int(0.2*cluster_feats.shape[0])
    print(len(idx_list),val_size)

    ### create folder for artifacts if it doesn't already exist
    artifacts_path = os.path.join(src_folder, 'analysis','artifacts')
    if os.path.exists(artifacts_path) == False:
        os.makedirs(artifacts_path)
    
    ### check if meta_type column exists in df_bvp and, if not, create it
    if 'meta_type' not in df_bvp.columns.values.tolist():
        # load metadata file
        try:
            df_meta = pd.read_csv('local_birds_meta.csv')
            df_meta.set_index('ID',inplace=True)
            print('reading metadata ...')
        except Exception as err:
            print('File read err for local_birds_meta.csv:', err)
            sys.exit()
            
        # add column for vocalization type to _bvp dataframe
        df_bvp['meta_type'] = ''
        
        # copy Rec_Content from metadata file to df_bvp
        for row in df_bvp.itertuples():
            df_bvp.at[row.Index,'meta_type'] = df_meta.at[int(row.src_file.replace('.mp3', '')), 'Rec_Content']

        # write df_bvp back to same filename as read
        df_bvp.to_csv(bvp_file, index=False)
        
    ### Setup the datasets
    # split dataset into training and validation randomly
    all_indices = np.array(idx_list)
    val_indices = np.random.choice(all_indices, size=val_size, replace=False)

    train_mask = np.isin(all_indices, val_indices, invert=True)
    train_indices = all_indices[train_mask]

    val_flat = cluster_feats[val_indices, :]
    train_flat = cluster_feats[train_indices, :]

    # scale and reshape
    for i in range(train_flat.shape[0]):
        train_flat[i,:] = train_flat[i,:]/np.max(train_flat[i,:])

    for i in range(val_flat.shape[0]):
        val_flat[i,:] = val_flat[i,:]/np.max(val_flat[i,:])

    train_images = np.reshape(train_flat, (-1,NUM_MELS,NUM_FRMS,1), order='F')
    test_images = np.reshape(val_flat, (-1,NUM_MELS,NUM_FRMS,1), order='F')

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(train_images.shape[0]).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_images.shape[0]).batch(batch_size))
                
    # instantiate the model
    model = CVAE(latent_dim,NUM_MELS,NUM_FRMS)
    
    # get one batch of data for initializing the model
    x = train_images[:batch_size]
    
    # the following call is needed to "define the forward path" which is required to save
    model(x)

    ### save the pre-training values of the model weigths for initializing each iteration
    model.save_weights(os.path.join(src_folder, 'analysis', 'models', 'init_weights'))
    
    df_discard = pd.DataFrame(columns=['src_file','segment_num'])
    df_discard [['src_file','segment_num']] = df_ids[['src_file','segment_num']]
    
    df_summary = pd.DataFrame(columns=['val_loss','cophenetic_corr','total_discards'])
    
    ### set iteration loop
    for iter in range(iterations):
        print('*********************** iteration {} ************************'.format(iter+1))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M-%S")
        iter_label = 'cvae_a2_z'+str(latent_dim)+'_'+timestamp
        print('timestamp =', timestamp,'iter_label=',iter_label)
        
        # load initial model weights for each model iteration
        model.load_weights(os.path.join(src_folder, 'analysis', 'models', 'init_weights'))
        
        ### Train the CVAE model
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for x in train_dataset:
                train_step(model, x, optimizer)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            for x in test_dataset:
                loss(compute_loss(model, x))
            elbo = -loss.result()

            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))

        # save the model weights
        model.save_weights(os.path.join(src_folder, 'analysis', 'models', 'cvae_a2_z'+str(latent_dim)+'_'+timestamp))

        ### get the latent space representation of all inputs

        input_flat = np.zeros((cluster_feats.shape[0], NUM_MELS * NUM_FRMS),dtype=np.float32)
        for i in range(cluster_feats.shape[0]):
            input_flat[i,:] = cluster_feats[i,:]/np.max(cluster_feats[i,:])

        input_img = np.reshape(input_flat, (-1,NUM_MELS,NUM_FRMS,1), order='F')

        # encode all input images
        z_mean, z_logvar = model.encode(input_img)

        df_ec = pd.DataFrame()
        df_ec['src_file'] = df_ids.src_file
        df_ec['segment_num'] = df_ids.segment_num

        for i in range(latent_dim):
            col = 'encoded_'+str(i)
            df_ec[col] = z_mean[:,i]

        ### Cluster latent space
        enc_cols = df_ec.columns.to_list()[2:]

        X = df_ec[enc_cols].to_numpy()
        df_ec['meta_type'] = ''

        # add meta_type field to df_ec
        for row in df_ec.itertuples():
            df_ec.at[row.Index, 'meta_type'] = df_bvp.loc[(df_bvp.src_file == row.src_file) & (df_bvp.segment_num == row.segment_num)].meta_type.to_string(index=False)

        """ 
        Compute discard recommendations based on model
        * designate as a "big cluster" any cluster whose size accounts for more than big_thresh
        * get cophenet distance between each cluster and each of the big clusters
        * clusters that are not close to a big cluster are more "anomalious" 
        so mark for discard clusters in order of distance with ties broken by smallest size first
        * mark clusters for discard until the max_discard threshold is reached
        """
        # generate the linkage matrix
        Z = linkage(X, 'average')

        c, coph_dists = cophenet(Z, pdist(X))
        
        # convert the coph_dist to squareform
        coph_sq = squareform(coph_dists)

        df_ec['cluster'] = fcluster(Z, n_clusters, criterion='maxclust')

        ### Compute encoding quality
        
        # cluster numbers and sizes 
        clusters = df_ec['cluster'].value_counts().to_dict()        
        
        # setup dataframe for metrics
        df_mets = pd.DataFrame.from_dict(clusters, orient='index', columns=['csize'])        
        
        # get dict of largest HAC clusters
        big_thresh = 0.05
        big_hac = {}
        tot_size = df_ec.shape[0]
        for c in clusters:
            if clusters[c] > big_thresh*tot_size:
                big_hac[c] = clusters[c]
                
        ### compute cophenetic distance between each cluster and the big clusters 
        ###    then take the min for each
        # get per cluster list of samples
        cluster_mems = []
        for c in range(n_clusters):
            cluster_mems.append(df_ec[df_ec['cluster']==c+1].index.values.tolist())

        dists = np.zeros((n_clusters,len(big_hac)))
        big_reps = []
        for b in big_hac:
            big_reps.append(cluster_mems[b-1][0])
            
        for c in range(n_clusters):
            rep = cluster_mems[c][0]
            for i in range(len(big_hac)):
                dists[c,i] = coph_sq[rep,big_reps[i]]
        
        # compute min distance to a big cluster for each cluster
        for c in range(n_clusters):
            df_mets.loc[c+1,'big_dist'] = min(dists[c,:])

        # get dictionary of cluster:csize sorted by [big_dist,csize]
        dist_dict = df_mets.sort_values(['big_dist','csize'],ascending=[False,True])['csize'].to_dict()

        ### compute per cluster discard recommendations based on csize then big_dist
        tot_disc = 0
        max_disc = int(max_disc_fract*df_ec.shape[0])

        df_mets['discard'] = 0
        for k in dist_dict:
            disc = dist_dict[k]
            if disc + tot_disc < max_disc:
                tot_disc += disc
                df_mets.loc[k,'discard'] = 1
            else:
                break

        # sort clusters by number of songs
        df_song = df_ec[df_ec.meta_type=='song']
        song_clusters = df_song['cluster'].value_counts().to_dict()
        song_sorted = sorted(song_clusters.items(), key=lambda x:x[1], reverse=True)

        if len(song_clusters) < n_clusters:
            missing_clusters = [x for x in list(clusters) if x not in list(song_clusters)]
            for key in missing_clusters:
                song_clusters[key] = 0
                
        # sort clusters by number of calls
        df_call = df_ec[df_ec.meta_type=='call']
        call_clusters = df_call['cluster'].value_counts().to_dict()
        call_sorted = sorted(call_clusters.items(), key=lambda x:x[1], reverse=True)

        if len(call_clusters) < n_clusters:
            missing_clusters = [x for x in list(clusters) if x not in list(call_clusters)]
            for key in missing_clusters:
                call_clusters[key] = 0
        
        for row in df_mets.itertuples():   
            df_mets.at[row.Index,'num_song'] = song_clusters[row.Index]
            df_mets.at[row.Index,'num_call'] = call_clusters[row.Index]
            df_mets.at[row.Index,'pct_song'] = float(df_mets.at[row.Index,'num_song']/df_mets.at[row.Index,'csize'])
            df_mets.at[row.Index,'pct_call'] = float(df_mets.at[row.Index,'num_call']/df_mets.at[row.Index,'csize'])
    
        ### Record per sample discard recommendations
        for row in df_discard.itertuples():
            cluster = df_ec.loc[row.Index,'cluster']
            df_discard.at[row.Index,iter_label] = df_mets.loc[cluster,'discard']
        
        ### Record per run summary stats 
        df_summary.at[iter_label,'cophenetic_corr'] = c
        df_summary.at[iter_label,'val_loss'] = elbo.numpy()
        df_summary.at[iter_label,'total_discards'] = df_mets.loc[(df_mets.discard==1),'csize'].sum()
        df_summary.at[iter_label,'num_big'] = len(big_hac)
        
        ### porting code from vade_clean.py 
        # generate the linkage matrix
        clusters = df_ec['cluster'].value_counts().to_dict()
        # get list of n_clusters largest clusters
        big_clusters = list(clusters.keys())[:n_clusters]
       
        # per-cluster metrics for summary
        for i in range(min(len(big_hac),num_big)):
            c = big_clusters[i]
            c_size = df_mets.at[c,'csize']
            df_summary.at[iter_label,'pct_size_'+str(i)] = c_size/len(df_ec.index)
            if song_call == 1:
                # compute song_call purity metric for each cluster
                num_song = df_mets.at[c,'num_song']
                num_call = df_mets.at[c,'num_call']
                df_summary.at[iter_label,'cpurity_'+str(i)] = (num_song-num_call)/c_size
        ### \porting code from vade_clean.py 

        '''
        compute hogeneity score excluding clusters recommended for discard
        '''
        disc_list = df_mets[df_mets.discard==1].index.to_list()
        df_nodisc = df_ec.drop(df_ec.loc[df_ec['cluster'].isin(disc_list)].index)
        df_nodisc.meta_type = pd.Categorical(df_nodisc.meta_type)
        labels = df_nodisc.meta_type.cat.codes.to_numpy()
        clusters = df_nodisc['cluster'].to_numpy()
        df_summary.at[iter_label,'homo_met_disc'] = homogeneity_score(labels, clusters)
        
        ### Save the metrics and clustered encodings
        df_mets.to_csv(os.path.join(artifacts_path,'metrics_'+iter_label+'.csv'), index=True)
        df_ec.to_csv(os.path.join(artifacts_path,'encoding_clustered_'+iter_label+'.csv'), index=False)
                    
        del df_mets
        del df_ec
    # end of iterations loop

    ### Save discard recommendations
    df_discard.to_csv(os.path.join(artifacts_path,'discard_'+iter_label+'.csv'), index=False)
    df_summary.to_csv(os.path.join(artifacts_path,'summary_'+iter_label+'.csv'), index=True)
main()