"""
vade_clean.py

Python script performs data UOD (i.e., cleaning) for the spectrogram dataset for one species using 
convolutional autoencoder (CAE) and Variational Deep Embedding (VaDE) as follows:

UOD using CAE:
* Dimensionality reduction: CAE is trained on preprocessed spectrograms clips from bird sound recordings. 
The parameters from model are also used as a starting point for VaDE model training (i.e., pre-training)
* Hierarchical agglomerative clustering (HAC) is applied to the CAE latent space representation 
of the clips to produce n_clusters
* Discard candidates are determined at the cluster level with smallest clusters farthest away from 
one of the biggest clusters recommended for discard first and proceeding to larger clusters until
the specified max discard percentage is reached
* The preceding process steps are repeated a specified number of times and each time the 
resulting model parameters, latent space encodings, cluster assignments and discard decisions are saved
* Final per clip discard decisions are made based on majority voting and these decisions
are saved.

UOD using VaDE:
* Train VaDE model on a one-species preprocessed spectrograms clip dataset using parameters from a CAE model trained 
on the same dataset as a starting point
* For each clip in the dataset calculate the highest probability of membership across the clusters identified
* Starting with lowest probability of cluster membership, designate clips to be candidate outliers until the total 
number of such candidates reaches the specified max number of discards

06-24-2025
- minor changes to move execution path down one level as part of folder restructuring for phase 2 and 
clean up header comments

02-10-2025
- added max_discards as argument (only impacts pretraining models since this is performed in 
vade_post.py for vade models)

01-28-2025
- understood and fixed the intermittent failure in pretraining evalution code. Problem was that no cluster 
satisfied big cluster threshold

01-27-2025
- fixed an intermittent failure in code evaluating pretraining models for n_clusters == 1. 
- Note: It's not understood why this problem only occurred intermittently since dists.shape is cx1
    df_mets.loc[c+1,'big_dist'] = min(dists[c,:])
ValueError: min() arg is an empty sequence

11-04-2024
- minor comment edits

10-10-2024 (rolled back because of issue with torch.utils.data.TensorDataset without labels)
- removed support for labels in dataset which were only used for plotting progress of training 
which is no longer being supported 

09-30-2024
- fixed bug where vade model files was written even when "failed to find requested number of clusters" 

09-26-2024
- added number of clusters found in output in "failed to find requested number of clusters" case

09-11-2024
- removed minor duplication of df_mets dataframe initialization
- corrected naming of pretrain_model filename to use iter_label

"""
import argparse

#import matplotlib.pyplot as plt
#from munkres import Munkres
#from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import homogeneity_score
import torch
import torch.nn.functional as F
import torch.utils.data
#from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from vade import AutoEncoderForPretrain, VaDE_CNN, lossfun, cluster_probs, _reparameterize

import os
import pathlib
import math
import time
from datetime import datetime
import glob
import numpy as np
import pandas as pd

import sys
from sys import argv

from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import pdist, squareform
import array

import warnings
warnings.filterwarnings('ignore')

PLOT_NUM_PER_CLASS = 128
#Z_DIM = 10
NUM_MELS = 32
NUM_FRMS = 40
T_BETA = 50
T_S = 50
BETA_SCHED_EXP = 5
BETA_0 = 0.1


def beta_sched(epoch, beta_0, T_beta, T_s, u):
    period = np.floor(epoch/(T_beta + T_s))
    return min(beta_0 + pow((epoch - period*(T_beta + T_s))/T_beta, u), 1.0)

def pretrain(model, data_loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    for x,_ in data_loader:
        batch_size = x.size(0)
        x = x.to(device)
        recon_x = model(x)
        loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Pretrain Epoch {:>3}: Train Loss = {:.4f}'.format(
        epoch, total_loss / len(data_loader)))
        
    return total_loss / len(data_loader)
    
def train(model, data_loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    total_recon = 0
    total_kl = 0
    for x,_ in data_loader:
        x = x.to(device)
        recon_x, mu, logvar = model(x)
        loss, recon_loss, kl_term = lossfun(model, x, recon_x, mu, logvar)
        total_loss += recon_loss.item() \
                    + beta_sched(epoch, BETA_0, T_BETA, T_S, BETA_SCHED_EXP)*kl_term.item()
        total_recon += recon_loss.item()
        total_kl += kl_term.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #writer.add_scalar('Loss/train', total_loss / len(data_loader), epoch)
    #print('Vade Epoch {:>3}: Train Loss = {:.4f} [{:.4f}, {:.4f}]'.format(
    #    epoch, total_loss / len(data_loader), total_recon / len(data_loader), total_kl / len(data_loader)))

    return total_loss / len(data_loader)

"""
function: test
Computes and returns the loss of the input model against the test dataset

In this version the tensorboard plotting has been removed!
"""
def test(model, data_loader, device, epoch):
    model.eval()

    #gain = torch.zeros((n_types, n_types), dtype=torch.int, device=device)
    with torch.no_grad():
        total_loss = 0
        for xs, ts in data_loader:
            xs, ts = xs.to(device).view(-1, 1,NUM_MELS,NUM_FRMS), ts.to(device)
            """
            ys = model.classify(xs)
            for t, y in zip(ts, ys):
                gain[t, y] += 1
            """    
            ### compute test loss
            recon_x, mu, logvar = model(xs)
            loss = lossfun(model, xs, recon_x, mu, logvar)[0]
            total_loss += loss.item()
            
        #cost = (torch.max(gain) - gain).cpu().numpy()
        #assign = Munkres().compute(cost)
        #acc = torch.sum(gain[tuple(zip(*assign))]).float() / torch.sum(gain)
        """
        # Plot latent space
        xs, ts = plot_points[0].to(device).view(-1, 1,NUM_MELS,NUM_FRMS), plot_points[1].numpy()
        zs = model.encode(xs)[0].cpu().numpy()
        tsne = TSNE(n_components=2, init='pca')
        zs_tsne = tsne.fit_transform(zs)

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("tab10")
        for t in range(n_types):
            points = zs_tsne[ts == t]
            ax.scatter(points[:, 0], points[:, 1], color=cmap(t), label=str(t))
        ax.legend()
        """
    #writer.add_scalar('Acc/test', acc.item(), epoch)
    #writer.add_scalar('Loss/test', total_loss / len(data_loader), epoch)
    
    #writer.add_figure('LatentSpace', fig, epoch)
    
    return total_loss / len(data_loader)

"""
function: bird_dataset

This function reads the files needed to create a bird sound spectrogram dataset 
for the specified bird species

Inputs:
- species 

Outputs:
- dataset (contains features and labels in pytorch format)
- features (needed for getting to latent space representation)
- BVP IDs (needed for song/call/other meta_type in summary stats)

"""
def bird_dataset(species):
    src_folder = '../dataset/audio/' + species

    ### read input files
    # get list of all _bvp files in analysis folder and select the latest one
    bvp_files_list = glob.glob(os.path.join(src_folder, 'analysis', species + '_bvp*.csv')) 
    bvp_file = max(bvp_files_list, key=os.path.getctime)
    
    df_bvp = pd.read_csv(bvp_file)
    print('BVP file:', bvp_file, ':', df_bvp.shape)
    
    # get list of all cluster_feats files in analysis folder and select the latest one
    cluster_files_list = glob.glob(os.path.join(src_folder, 'analysis', 'feats_32x40_*.csv')) 
    if len(cluster_files_list) == 0:
        print('\nNo 32x40 features files found for', species,'!!!')
        sys.exit('Error Exit!')
        
    cluster_feat_file = max(cluster_files_list, key=os.path.getmtime)

    # load cluster_feat file
    cluster_feats = np.loadtxt(cluster_feat_file, delimiter=',', dtype='float32')
    print('cluster_feats file:', cluster_feat_file, ':', cluster_feats.shape)
        
    # get list of all bvp_id files in analysis folder and select the latest one
    bvp_ids_files_list = glob.glob(os.path.join(src_folder, 'analysis', 'bvp_ids*.csv')) 
    bvp_ids_file = max(bvp_ids_files_list, key=os.path.getctime)

    # load src_file and BVP details from the bvp_ids file that match cluster_feat.csv
    df_ids = pd.read_csv(bvp_ids_file)
    print('BVP IDs file:', bvp_ids_file, ':', df_ids.shape)

    ### Setup the dataset
    input_imgs = torch.from_numpy(np.reshape(cluster_feats, (-1,1,NUM_MELS,NUM_FRMS), order='F'))

    # get labels for song/call/other summary stats
    df_ids['meta_type'] = ''
    for row in df_ids.itertuples():
        df_ids.at[row.Index,'meta_type'] = df_bvp[(df_bvp.src_file==row.src_file) 
                                                 & (df_bvp.segment_num==row.segment_num)].meta_type.values[0]
    
    labels = np.zeros(cluster_feats.shape[0],dtype=np.int8)
    class_dict = {'song':0, 'call':1, 'both':2, 'other':3}
    for row in df_ids.itertuples():
        labels[row.Index] = class_dict[df_ids.loc[row.Index,'meta_type']]
    input_lbls = torch.from_numpy(labels)
    
    print(input_imgs.shape, input_lbls.shape)
    dataset = torch.utils.data.TensorDataset(input_imgs, input_lbls)

    return dataset, cluster_feats, df_ids  

    
def main():
    parser = argparse.ArgumentParser(
        description='Train VaDE with one species bird sounds dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to process BVPs for',
                        type=str)
    parser.add_argument('--pretrain-epochs', '-p',
                        help='Number of epochs.',
                        type=int, default=20)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs.',
                        type=int, default=20)
    parser.add_argument('--gpu', '-g',
                        help='GPU id. (Negative number indicates CPU)',
                        type=int, default=-1)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning Rate.',
                        type=float, default=0.001)
    parser.add_argument('--batch-size', '-b',
                        help='Batch size.',
                        type=int, default=128)
    parser.add_argument('--n-clusters', '-c',
                        help='Number of classes.',
                        type=int, default=2)
    parser.add_argument('--n-types', '-t',
                        help='Number of label values (types).',
                        type=int, default=4)                       
    parser.add_argument('--z-dim', '-z', 
                        help='latent variable dimension',
                        type=int, default=10)
    parser.add_argument('--iterations', '-i',
                        help='number of models to train and evaluate',
                        type=int, default=11)
    parser.add_argument('--song-call', '-s',
                        help='does species have distinct song and call (1=yes), 0=no)',
                        type=int, default=1)
    parser.add_argument('--max-discards', '-d',
                        help='maximum fraction of BVPs to discard (default: 0.1)',
                        type=float, default=0.1)
    args = parser.parse_args()


#    parser.add_argument('--max-discards', type=float, default=0.1, help='maximum fraction of BVPs to discard (default: 0.1')

    args = parser.parse_args()
    species = args.species
    z_dim = args.z_dim
    iterations = args.iterations
    n_clusters = args.n_clusters
    pre_epochs = args.pretrain_epochs
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    song_call = args.song_call

    # parameters for pretrained model discard recommendations
    hac_clusters = 50
    big_thresh_pct = 0.05
    max_disc_fract = args.max_discards
    
    print('\n ************************** Run Arguments ****************************\n')
    print('bird species:',species,'\n latent dim=',z_dim,'\n iterations=',iterations,'\n number of clusters=',n_clusters)
    print(' pre_epochs=',pre_epochs,'\n vade epochs=',epochs,'\n batch_size=',batch_size,'\n learning rate=',lr)
    print(' song call=',song_call)
    if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device('cuda:{}'.format(args.gpu) if if_use_cuda else 'cpu')
    
    dataset, cluster_feats, df_ids = bird_dataset(args.species)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=if_use_cuda)

    ### create folder for artifacts if it doesn't already exist
    src_folder = '../dataset/audio/' + args.species
    artifacts_path = os.path.join(src_folder, 'analysis','artifacts')
    if os.path.exists(artifacts_path) == False:
        os.makedirs(artifacts_path)
    
    # dataframe for per-sample anomalousness scores
    df_anomal = pd.DataFrame(columns=['src_file','segment_num'])
    df_anomal [['src_file','segment_num']] = df_ids[['src_file','segment_num']]
    
    df_summary = pd.DataFrame(columns=['pretrain_loss','train_loss','test_loss'])
    clustering = np.zeros(df_ids.shape[0], dtype=np.int8)
    labels = df_ids['meta_type'].to_numpy()
    
    ### setup iteration loop
    iter = 0
    while iter < iterations:
        print('*********************** iteration {} ************************'.format(iter+1))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M-%S")
        iter_label = 'c'+str(n_clusters)+'_z'+str(z_dim)+'_'+timestamp
        print('timestamp =', timestamp,'iter_label=',iter_label)
        
        pretrain_model = AutoEncoderForPretrain(NUM_MELS, NUM_FRMS, z_dim).to(device)

        optimizer = torch.optim.Adam(pretrain_model.parameters(),
                                     lr=args.learning_rate)
        
        ### train pretrain model 
        for epoch in range(1, pre_epochs + 1):
            pretrain_loss = pretrain(pretrain_model, data_loader, optimizer, device, epoch)

        with torch.no_grad():
            x = torch.cat([data[0] for data in dataset]).view(-1,1,NUM_MELS,NUM_FRMS).to(device)
            z = pretrain_model.encode(x).cpu()
            
        pretrain_model = pretrain_model.cpu()
        state_dict = pretrain_model.state_dict()

        gmm = GaussianMixture(n_components=args.n_clusters, covariance_type='diag')
        gmm.fit(z)

        ### To consider: pretrain model characterization and/or selection???
        ### save pretrained model 
        torch.save(pretrain_model.state_dict(), os.path.join(src_folder, 'analysis', 'models', 'pretrained_model_'+iter_label+'.pth'))

        ### Summary of pretrain model
        df_summary.at[iter_label,'pretrain_loss'] = pretrain_loss
        
        # compute metrics for pretrain encodings clustered
        df_ec = pd.DataFrame()
        df_ec[['src_file','segment_num','meta_type']] = df_ids[['src_file','segment_num','meta_type']]
        
        ### HAC 
        # generate the linkage matrix
        link_mat = linkage(z, 'average')
        df_ec['hcluster(12)'] = fcluster(link_mat, 12, criterion='maxclust')
        clusters = df_ec['hcluster(12)'].value_counts().to_dict()
        # get list of n_clusters largest clusters
        big_clusters = list(clusters.keys())[:n_clusters]
       
        # per-cluster metrics
        for i in range(n_clusters):
            c = big_clusters[i]
            c_size = len(df_ec[df_ec['hcluster(12)']==c].index)
            df_summary.at[iter_label,'pre_size_'+str(i)] = c_size/len(df_ec.index)
            if song_call == 1:
                # compute song_call purity metric for each cluster
                df_clust = df_ec[df_ec['hcluster(12)']==c]
                num_song = df_clust[df_clust['meta_type']=='song'].src_file.count()
                num_call = df_clust[df_clust['meta_type']=='call'].src_file.count()
                df_summary.at[iter_label,'pre_purity_'+str(i)] = (num_song-num_call)/c_size
        
        """ 
        Compute discard recommendations based on pretrained model
        * designate as a "big cluster" any cluster whose size accounts for more than big_thresh
        * get cophenet distance between each cluster and each of the big clusters
        * clusters that are not close to a big cluster are more "anomalious" 
        so mark for discard clusters in order of distance with ties broken by smallest size first
        * mark clusters for discard until the max_discard threshold is reached
        """
        ### Compute encoding clustered quality metrics
        dist_mat = pdist(z)
        pdist_sq = squareform(dist_mat)

        c, coph_dists = cophenet(link_mat, dist_mat)
        coph_sq = squareform(coph_dists)
        
        df_ec['hac'] = fcluster(link_mat, hac_clusters, criterion='maxclust')
        clusters = df_ec['hac'].value_counts().to_dict()
        # get dict of largest HAC clusters
        big_hac = {}
        big_thresh = df_ec.shape[0]*big_thresh_pct
        for c in clusters:
            if clusters[c] >= big_thresh:
                big_hac[c] = clusters[c]
                
        # if no HAC cluster exceeds the size threshold to be _big_ then set the biggest cluster as big_hac
        if len(big_hac) == 0:
            biggest_cluster = max(clusters, key=clusters.get)
            big_hac[biggest_cluster] = clusters[biggest_cluster]
        
        df_mets = pd.DataFrame.from_dict(clusters, orient='index', columns=['size'])

        ### compute cophenetic distance between each cluster and the big clusters 
        ###    then take the min for each
        # get per cluster list of samples
        cluster_mems = []
        for c in range(hac_clusters):
            cluster_mems.append(df_ec[df_ec['hac']==c+1].index.values.tolist())

        dists = np.zeros((hac_clusters,len(big_hac)))
        big_reps = []
        for b in big_hac:
            big_reps.append(cluster_mems[b-1][0])
            
        for c in range(hac_clusters):
            rep = cluster_mems[c][0]
            for i in range(len(big_hac)):
                dists[c,i] = coph_sq[rep,big_reps[i]]
        
        # compute min distance to a big cluster for each cluster for n_clusters > 1
        for c in range(hac_clusters):
            if n_clusters > 1:
                try:
                    df_mets.loc[c+1,'big_dist'] = min(dists[c,:])
                except Exception as err:
                    print(c, dists[c,:])
                    continue
            else:   # n_clusters == 1 so dists[c] is the min distance
                df_mets.loc[c+1,'big_dist'] = dists[c,0]
                
        # get dictionary of cluster:size sorted by [big_dist,size]
        dist_dict = df_mets.sort_values(['big_dist','size'],ascending=[False,True])['size'].to_dict()

        tot_disc = 0
        max_disc = int(max_disc_fract*df_ec.shape[0])

        ### compute per cluster discard recommendations based on size then big_dist
        cluster_disc = array.array('I',(0 for i in range(0,hac_clusters)))
        for k in dist_dict:
            disc = dist_dict[k]
            if disc + tot_disc < max_disc:
                tot_disc += disc
                cluster_disc[k-1] =  1
            else:
                break
        
        ### distribute discard recommendations to samples
        discard_recs = array.array('I',(0 for i in range(0,df_ec.shape[0])))
        for i in range(0,df_ec.shape[0]):
            c = df_ec.loc[i,'hac']
            discard_recs[i] = cluster_disc[c-1]

        ### save pretraining per-sample discard recommendations for each iteration
        df_anomal ['pre'+iter_label[-17:]] = discard_recs
        
        # create VADE model
        model = VaDE_CNN(args.n_clusters, NUM_MELS, NUM_FRMS, z_dim)

        # transfer pretrain model weights to VADE model
        model.load_state_dict(state_dict, strict=False)
        model._pi.data = torch.log(torch.from_numpy(gmm.weights_)).float()
        model.mu.data = torch.from_numpy(gmm.means_).float()
        model.logvar.data = torch.log(torch.from_numpy(gmm.covariances_)).float()
        model = model.to(device)
        
        ### train VADE model
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            pin_memory=if_use_cuda)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True,
            pin_memory=if_use_cuda)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # LR decreases every 10 epochs with a decay rate of 0.9
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.9)
        
        train_fail = False
        for epoch in range(1, args.epochs + 1):
            try:
                train_loss = train(model, train_loader, optimizer, device, epoch)
                test_loss = test(model, test_loader, device, epoch)
                print('Vade Epoch {:>3}: Train Loss = {:.4f} Val Loss = {:.4f}'.format(
                    epoch, train_loss, test_loss))
            except:
                train_fail = True
                break
                
            lr_scheduler.step()

        # continue to next iteration if train_fail occurs
        if train_fail:
            print('Training failure occurred in iteration {}!!! Cleanup and exit'.format(iter+1))
            # save summary and anomal results if any
            if iter > 0:
                df_anomal.to_csv(os.path.join(artifacts_path,'anomal_'+iter_label+'.csv'), index=False)
                df_summary.to_csv(os.path.join(artifacts_path,'summary_'+iter_label+'.csv'), index=True)
                sys.exit('Exit after training failure')
            continue
            
        ### output latent space encodings
        model = model.to('cpu')
        # encode all input images
        with torch.no_grad():
            z_mean, z_logvar = model.encode(torch.from_numpy(np.reshape(cluster_feats, (-1,1,NUM_MELS,NUM_FRMS), order='F')))
        
        # add cluster assignments 
        mu_priors = model.mu
        logvar_priors = model.logvar
        #z = _reparameterize(z_mean, z_logvar)
        #log_p_z_given_c = cluster_probs(z, mu_priors, logvar_priors)
        log_p_z_given_c = cluster_probs(z_mean, mu_priors, logvar_priors)

        df_ec = pd.DataFrame()
        df_ec[['src_file','segment_num','meta_type']] = df_ids[['src_file','segment_num','meta_type']]

        # get arrays ready to save
        with torch.no_grad():
            z_mean = z_mean.cpu().numpy()
            z_logvar = z_logvar.cpu().numpy()
            log_p_z_given_c.cpu().numpy()
        
        for i in range(z_dim):
            col = 'encoded_'+str(i)
            df_ec[col] = z_mean[:,i]

        clust_cols = []
        for i in range(n_clusters):
            clust = 'cluster_'+str(i)
            clust_cols.append(clust)
            with torch.no_grad():
                df_ec[clust] = log_p_z_given_c[:,i]
        
        df_ec['cluster_assign'] = df_ec[clust_cols].idxmax(axis='columns')
        df_ec.cluster_assign = pd.Categorical(df_ec.cluster_assign)
        clustering = df_ec.cluster_assign.cat.codes.to_numpy()

        df_ec['max_prob'] = 0.0
        for row in df_ec.itertuples():
            df_ec.at[row.Index, 'max_prob'] = np.exp(df_ec.loc[row.Index,row.cluster_assign])

        clusters = df_ec['cluster_assign'].value_counts().to_dict()
        # get list clusters sorted by size
        big_clusters = list(clusters.keys())
        
        if len(big_clusters) < n_clusters:
            print('requested {} clusters but model only found {}! Discard and restart iteration {}'.format(n_clusters,len(big_clusters),iter+1))
            del df_ec
            continue
        else:
            ### save trained model 
            torch.save(model.state_dict(), os.path.join(src_folder, 'analysis', 'models', 'vade_model_'+iter_label+'.pth'))

            ### Save the clustered encodings  
            df_ec.to_csv(os.path.join(artifacts_path,'vade_ec_'+iter_label+'.csv'), index=False)

            ### Save anomalousness scores for each iteration
            df_anomal [iter_label] = df_ec['max_prob']
            
            ### Summary stats for run 
            df_summary.at[iter_label,'train_loss'] = train_loss
            df_summary.at[iter_label,'test_loss'] = test_loss

            # per-cluster metrics
            for i in range(n_clusters):
                c = big_clusters[i]
                c_size = len(df_ec[df_ec['cluster_assign']==c].index)
                df_summary.at[iter_label,'vade_size_'+str(i)] = c_size/len(df_ec.index)
                if song_call == 1:
                    # compute song_call purity metric for each cluster
                    df_clust = df_ec[df_ec['cluster_assign']==c]
                    num_song = df_clust[df_clust['meta_type']=='song'].src_file.count()
                    num_call = df_clust[df_clust['meta_type']=='call'].src_file.count()
                    df_summary.at[iter_label,'vade_purity_'+str(i)] = (num_song-num_call)/c_size
            
            if song_call:
                df_summary.at[iter_label,'homo_met'] = homogeneity_score(labels, clustering)
            del df_ec
            
        iter += 1
    # end of iterations loop

    ### Save anomalousness scores
    df_anomal.to_csv(os.path.join(artifacts_path,'anomal_'+iter_label+'.csv'), index=False)
    df_summary.to_csv(os.path.join(artifacts_path,'summary_'+iter_label+'.csv'), index=True)
main()