import numpy as np
import pandas as pd
import xgboost as xgb
import time



def calculate_weights(labels):
    label_counts = labels.value_counts()
    weights = list()
    for k in labels:
        w = label_counts.sum()/label_counts[k]
        weights.append(w)
    weights = np.asarray(weights)
    return(weights)


if __name__ == '__main__':
    
    
    max_depth_levels = [2,3,4,5,6,7,8]
    n_rounds = 500
    data_for_xgb = pd.read_csv('/lustre/scratch/kiviaho/myoma/myoma-new/xgboost-averaging/features_for_xgboost.csv')
    slides_per_folds = pd.read_csv('/lustre/scratch/kiviaho/myoma/myoma-new/xgboost-averaging/slide_fold_information.csv',index_col=0)
    slides_per_folds = slides_per_folds.sort_values(by=['Sample']).reset_index(drop=True)
    data_for_xgb = data_for_xgb.sort_values(by=['sample']).reset_index(drop=True)

    
    # Create the splits for cv:
    if all(data_for_xgb['sample'] == slides_per_folds['Sample']):
        folds = ['fold_1','fold_2','fold_3','fold_4','fold_5']
        fold_indices = list()
        for f in folds:
            idx_in = list(np.where(slides_per_folds['Fold'] == f)[0])
            idx_out = list(np.where(slides_per_folds['Fold'] != f)[0])
            fold_indices.append((idx_in,idx_out))

    train_data = np.asarray(data_for_xgb.drop(columns=['sample','label','patches']))
    train_meta = data_for_xgb[['sample','label','patches']]
    train_labels = train_meta['label'].astype('category')
    weights = calculate_weights(data_for_xgb['label'])
    dtrain = xgb.DMatrix(train_data,label=train_labels,weight=weights)



    mean_auc_df = pd.DataFrame()
    mlogloss_df = pd.DataFrame()

    for depth in max_depth_levels:
        params = {'max_depth':depth,
              'objective':'multi:softprob',
              'num_class':6,
              'verbosity':0,
              'eval_metric':['mlogloss','auc'],
                  'tree_method':'gpu_hist'}
        
        # Calculate cross-validation-results
        cross_val_results = xgb.cv(params,
                                   dtrain,
                                   nfold=5,
                                   folds=fold_indices,
                                   num_boost_round=n_rounds,)
        
        mlogloss_df = pd.concat([mlogloss_df,cross_val_results['test-mlogloss-mean']],axis=1,ignore_index=True)
        mean_auc_df = pd.concat([mean_auc_df,cross_val_results['test-auc-mean']],axis=1,ignore_index=True)
        print('Cross validation using max depth='+str(depth)+' done.')
        cross_val_results.to_csv('/lustre/scratch/kiviaho/myoma/myoma-new/xgboost-averaging/xgb_cv_res_max_depth_'+str(depth)+'.csv')

    mean_auc_df.columns = max_depth_levels
    mlogloss_df.columns = max_depth_levels

    mean_auc_df.index = mean_auc_df.index+1
    mlogloss_df.index = mlogloss_df.index+1


    mlogloss_df.to_csv('/lustre/scratch/kiviaho/myoma/myoma-new/xgboost-averaging/mlogloss-cross-validation-results-same-split.csv')
    mean_auc_df.to_csv('/lustre/scratch/kiviaho/myoma/myoma-new/xgboost-averaging/mean-auc-cross-validation-results-same-split.csv')

