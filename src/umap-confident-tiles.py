# Date: 26.7.2022
# Author: Antti Kiviaho
#
# Cluster confidently predicted tiles to find common features for interpretation.

from matplotlib import pyplot as plt
import seaborn as sns
from itertools import cycle

import numpy as np
import pandas as pd
import umap
import hdbscan
from numpy.random import default_rng

import warnings
warnings.filterwarnings('ignore')

import time
import os
import sys

from enum import IntEnum
class toIntEnum(IntEnum):
    MED12 = 0
    HMGA2 = 1
    UNK = 2
    HMGA1 = 3
    OM = 4
    YEATS4 = 5

n_subsample = 400000
n_neighbors = 30
min_dist = 0.0
metric = 'cosine'
n_dims = 20
    
#########

which_fold = str(sys.argv[1])
metadata = pd.read_csv('/lustre/scratch/kiviaho/myoma/myoma-new/conv2d_93_layers/'+which_fold+'_confidently_predicted.tsv',sep='\t')
data = np.load('/lustre/scratch/kiviaho/myoma/myoma-new/conv2d_93_layers/'+which_fold+'_conv2d_93_layers.npy')

#Sample the data to fit into memory

rng = default_rng(seed=42)
idxs = rng.choice(len(data), size=n_subsample, replace=False)
data = data[idxs]
metadata = metadata.iloc[idxs].reset_index(drop=True)

save_path = '/lustre/scratch/kiviaho/myoma/myoma-new/subclass-features/umap/'

target_names = ['MED12', 'HMGA2', 'UNK', 'HMGA1', 'OM', 'YEATS4']
n_classes = len(target_names)
colors = cycle(["aqua", "darkorange", "cornflowerblue",'violet','red','gold'])

########

# STANDARD FOR VISUALIZATION
print('Calculating standard UMAP for visualization...')
standard_embedding = umap.UMAP(random_state=42, low_memory = True).fit_transform(data)


######## 
# CLUSTERING
print('Calculating UMAP for clustering...')
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_dims, metric=metric, low_memory = True)

data_umap = reducer.fit_transform(data)

print('Running HDBSCAN on UMAP...')
labels = hdbscan.HDBSCAN(
    min_samples=10,
    min_cluster_size=500,
).fit_predict(data_umap)

#######
print('Plotting and saving...')

fig, (ax1, ax2) = plt.subplots(1, 2)

clustered = (labels >= 0)
ax1.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            color=(0.5, 0.5, 0.5),
            s=0.7,
            alpha=0.7)
scatter = ax1.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=labels[clustered],
            s=0.7,
            cmap='Spectral');

legend1 = ax1.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax1.add_artist(legend1)

for i, color in zip(range(n_classes), colors):
    ax2.scatter(
        standard_embedding[metadata[metadata['Label']==i].index][:,0],
        standard_embedding[metadata[metadata['Label']==i].index][:,1],
        color=color,
        label=target_names[i],
        s=0.7,
        alpha=0.7
    )
ax1.set_title('HDBSCAN clustering of UMAP embedding')
ax2.set_title('Original labels on UMAP embedding')
ax2.legend(loc="lower right",prop={'size': 15},markerscale=20)
fig.set_figheight(20)
fig.set_figwidth(40)
plt.tight_layout()
fig.suptitle(which_fold + ' side-by-side UMAP')

plt.tight_layout()
plt.savefig(save_path+'HDBSCAN_on_UMAP_'+str(n_dims)+'_dimensions_'+ which_fold +'_'+ str(n_neighbors)+ '_neighbors_'+ str(min_dist) +'_'+metric+ '_distance.png')

np.save(save_path+'HDBSCAN_on_UMAP_labels_'+str(n_dims)+'_dimensions_'+ which_fold +'_'+ str(n_neighbors)+ '_neighbors_'+ str(min_dist) +'_'+metric+ '_distance.npy',labels)

np.save(save_path+'UMAP_visualization_data_2_dims'+ which_fold +'_'+ str(n_neighbors)+ '_neighbors_'+ str(min_dist) +'_'+metric+ '_distance_dimensions.npy',standard_embedding)

np.save(save_path+'UMAP_clustering_data_'+str(n_dims) +'_dims_' + which_fold +'_'+ str(n_neighbors)+ '_neighbors_'+ str(min_dist) +'_'+metric+ '_distance_dimensions.npy',data_umap)

