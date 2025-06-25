"""
scatterplot.py

Python script displays a scatter plot of a 2D encoding saved in a column format
with headings named "encoded_" followed by a number between 0 and z_dim-1.
If z_dim is > 2 then t-SNE is applied.

06-25-2025
- minor changes to move execution path down one level as part of folder restructuring for phase 2

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

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main():
    parser = argparse.ArgumentParser(
        description='Scatter plot 2D latent space',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('species',  
                        help='bird species to process BVPs for',
                        type=str)
    parser.add_argument('filename',
                        help='name of .csv file containing encodings',
                        type=str)
    parser.add_argument('--color-col', '-c',
                        help='column name to assign colors with',
                        type=str, default='meta_type')
    parser.add_argument('--point-sz', '-s',
                        help='size of marker in scatter plot',
                        type=int, default=14)
    parser.add_argument('--z-dim', '-z',
                        help='input dimensions',
                        type=int, default=10)

    args = parser.parse_args()
    species = args.species
    filename = args.filename
    color_col = args.color_col
    point_sz = args.point_sz
    z_dim = args.z_dim

    # load encodings file assumed to be in the artifacts folder under the species input
    src_dir = pathlib.Path('../dataset/audio')
    src_folder = os.path.join(src_dir, species)
    try:
        df_enc = pd.read_csv(os.path.join(src_folder, 'analysis','artifacts',filename))
        print('processing ', os.path.join(src_folder, 'analysis','artifacts',filename))
    except Exception as err:
        print('File read err for ', os.path.join(src_folder, 'analysis','artifacts',filename), ':', err)
        sys.exit()
        

    cols = []
    for i in range(z_dim):
        cols.append('encoded_'+str(i))

    if z_dim > 2:
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(df_enc[cols])

        df_enc['tsne_0'] = tsne_results[:,0]
        df_enc['tsne_1'] = tsne_results[:,1]

    # scatter plot and color code encoded by color_col
    grps = df_enc[color_col].value_counts().to_dict()
    grp_names = list(grps.keys())
    plt.figure(figsize=(10, 10))
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(grp_names))))

    for grp in grp_names:
        df_c = df_enc[df_enc[color_col]==grp]
        if z_dim > 2:
            plt.scatter(df_c.tsne_0,df_c.tsne_1,color=next(color),s=point_sz,marker='.',label=grp)
        else:
            plt.scatter(df_c.encoded_0,df_c.encoded_1,color=next(color),s=point_sz,marker='.',label=grp)
    plt.legend(title=color_col)
    plt.grid(color = 'y')
    plt.show()

main()