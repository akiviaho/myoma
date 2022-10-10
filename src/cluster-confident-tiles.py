# Date: 26.7.2022
# Author: Antti Kiviaho
#
# Cluster confidently predicted tiles to find common features for interpretation.

from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from itertools import cycle

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


if __name__ == '__main__':
    ############# INPUT VARIABLES / FILES ###########
    metadata = pd.read_csv('/lustre/scratch/kiviaho/myoma/myoma-new/conv2d_93_layers/50k_fold_5_confidently_predicted.tsv',sep='\t')

    data = np.load('/lustre/scratch/kiviaho/myoma/myoma-new/conv2d_93_layers/50k_fold_5_conv2d_93_layers.npy',)
    metadata = metadata[:data.shape[0]]

    save_path = '/lustre/scratch/kiviaho/myoma/myoma-new/conv2d_93_layers/'

    n_kmean_clusters = range(10,15)


    #############

    #Transform the data PCA --> tSNE
    pca = PCA(50)
    data_pcs = pca.fit_transform(data)
    data_tsne = TSNE(n_components=2,init='random').fit_transform(data_pcs)

    # Plot tSNE with different kmeans clusters
    for n_clusters in n_kmean_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_tsne)
        metadata['cluster'] = kmeans.labels_

        plt.figure(figsize=(15,15),dpi=200)
        colors = cm.rainbow(np.linspace(0, 1, n_clusters))
        for i, color in zip(range(n_clusters), colors):
            plt.scatter(
                data_tsne[metadata[metadata['cluster']==i].index][:,0],
                data_tsne[metadata[metadata['cluster']==i].index][:,1],
                color=color,
                label=i
            )

        plt.title("TSNE plot of confidently predicted fold 5 tiles",size=15)
        plt.legend(loc="lower right",prop={'size': 15})
        plt.tight_layout()
        plt.savefig(save_path+'TSNE_kmeans_'+str(n_clusters)+'_fold_5_confident_subset.png')



    # Plot tSNE with sample categories
    target_names = ['MED12', 'HMGA2', 'UNK', 'HMGA1', 'OM', 'YEATS4']
    n_classes = len(target_names)

    plt.figure(figsize=(15,15),dpi=200)
    colors = cycle(["aqua", "darkorange", "cornflowerblue",'violet','red','gold'])
    for i, color in zip(range(n_classes), colors):
        plt.scatter(
            data_tsne[metadata[metadata['Label']==i].index][:,0],
            data_tsne[metadata[metadata['Label']==i].index][:,1],
            color=color,
            label=target_names[i]
        )

    plt.title("TSNE plot of confidently predicted fold 5 tiles",size=15)
    plt.legend(loc="lower right",prop={'size': 15})
    plt.tight_layout()
    plt.savefig(save_path+'TSNE_sample_types_fold_5_confident_subset.png')

