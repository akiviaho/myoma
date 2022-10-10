import os
import sys
from logzero import logger
from matplotlib import pyplot as plt
from itertools import cycle
from scipy import stats
from collections import Counter
import xgboost as xgb

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from enum import IntEnum

from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator

from keras.models import load_model

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class toIntEnum(IntEnum):
    MED12 = 0
    HMGA2 = 1
    UNK = 2
    HMGA1 = 3
    OM = 4
    YEATS4 = 5

def download_results(sheet_list,folds):
    
    results = pd.DataFrame()
    for sheet,which_fold in zip(sheet_list,folds):
        fold_res = pd.read_csv('/lustre/scratch/kiviaho/myoma/myoma-new/'+sheet)
        tfrecord_contents = pd.read_csv('/lustre/scratch/kiviaho/myoma/myoma-new/'+which_fold+'_tfr_contents.tsv')
        tfrecord_contents = tfrecord_contents[:len(fold_res)]

        fold_res = fold_res.rename(columns={'0': 0, '1': 1,'2': 2,'3': 3,'4': 4,'5': 5})
        fold_res = pd.concat([tfrecord_contents.reset_index(drop=True), fold_res], axis=1)
        results = pd.concat([fold_res.reset_index(drop=True), results.reset_index(drop=True)])
        results.dropna(inplace=True)
    return results
    

def calculate_features_for_xgboost(data):
    samples = np.unique(data['Sample'])

    all_features = pd.DataFrame()
    for sample in samples:
        df = data[data['Sample'] == sample].reset_index(drop=True)
        lab = df['Label'][0].astype(int)
        preds = df[[0,1,2,3,4,5]]
        max_idx = preds.idxmax(axis=1)

        slide_features = pd.DataFrame({'sample':[sample],
                                    'label': [lab],
                                    'patches': len(df)})
        for col in preds:
            vals = df[col]

            features = pd.DataFrame({'n top predicted':[(max_idx==col).sum()],
                         'sum':[np.sum(vals)],
                         'median':[np.median(vals)],
                         'max':[np.max(vals)],
                         '99.75th percentile':[np.percentile(vals,99.75)],
                         '99.5th percentile':[np.percentile(vals,99.5)],
                         '99th percentile':[np.percentile(vals,99)],
                         '98th percentile':[np.percentile(vals,98)],
                         '95th percentile':[np.percentile(vals,95)],
                         '90th percentile':[np.percentile(vals,90)],
                         '80th percentile':[np.percentile(vals,80)],
                         '10th percentile':[np.percentile(vals,10)],
                         'n samples with prob > 0.999':[(vals>0.999).sum()],
                         'n samples with prob > 0.99':[(vals>0.99).sum()],
                         'n samples with prob > 0.9':[(vals>0.9).sum()]})
            fixed_names = [toIntEnum(col).name + ' ' + c for c in features.columns]
            features.set_axis(fixed_names, axis=1, inplace=True)

            slide_features = pd.concat([slide_features, features],axis=1)
        all_features = pd.concat([all_features, slide_features],axis=0,ignore_index=True)

    return(all_features)
    
if __name__ == "__main__":

    
    folds = ['fold_1','fold_2','fold_3','fold_4','fold_5']
    sheets = ['multiclass_'+folds[0]+'_missing_2_epochs_100_percent_21939635_at_2022-07-15_21:45:39_598px_multiclass_prediction_results.csv',
                 'multiclass_'+folds[1]+'_missing_2_epochs_100_percent_21939634_at_2022-07-15_12:05:59_598px_multiclass_prediction_results.csv',
                 'multiclass_'+folds[2]+'_missing_2_epochs_100_percent_21939633_at_2022-07-15_14:57:30_598px_multiclass_prediction_results.csv',
                 'multiclass_'+folds[3]+'_missing_2_epochs_100_percent_21939632_at_2022-07-15_16:04:26_598px_multiclass_prediction_results.csv',
                 'multiclass_'+folds[4]+'_missing_2_epochs_100_percent_21892820_at_2022-07-04_09:01:38_598px_multiclass_prediction_results.csv']

    all_folds = download_results(sheets,folds)

    xgb_features = calculate_features_for_xgboost(all_folds)

    xgb_features.to_csv('/lustre/scratch/kiviaho/myoma/myoma-new/xgboost-averaging/features_for_xgboost.csv',index=False)

