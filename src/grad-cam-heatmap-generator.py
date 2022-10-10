# Author: Antti Kiviaho
# Date: 9.6.2022
#
# Grad-CAM environment for finding the most important features affecting CNN function 
# https://keras.io/examples/vision/grad_cam/
#
#
# Example run: python -u /lustre/scratch/kiviaho/myoma/myoma-new/src/grad-cam-heatmap-generator.py MED12 2000
# Where MED12 is the section type is what we want to look at and 2000 the number of samples

import sys
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from collections import Counter

# FUNCTION DEFINITIONS
from enum import IntEnum
class toIntEnum(IntEnum):
    wt = 0
    MED12 = 1
    HMGA2 = 2
    YEATS4 = 3

def get_coordinates(df_subset):
    # Getting the coordinates
    tiles = [s.split('/')[len(s.split('/'))-1].strip('.jpg') for s in df_subset['Path']]
    tile_coords = []
    for s in tiles:
        if 'm' in s:
            tile_coords.append(s.split('_')[2:4])
        elif 'T' in s:
            tile_coords.append(s.split('_')[4:6])
    tile_coords = np.asarray(tile_coords,dtype=np.int64)
    df_subset['y_coord'] = tile_coords[:,1]
    df_subset['x_coord'] = tile_coords[:,0]
    df_subset.sort_values(['x_coord', 'y_coord'], ascending=[True, False],inplace=True)
    df_subset = df_subset.reset_index(drop=True)
    return df_subset

def get_img_array(img_path, size=(598,598,3)):
    # `img` is a PIL image of size 598 x 598 x 3
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (598, 598, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 598, 598, 3)
    array = np.expand_dims(array, axis=0)
    array = np.divide(array,255)
    return array


def make_gradcam_heatmap(img_array, model, grad_model, last_conv_layer_name, pred_index=None):
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# DOWNLOAD THE SAMPLE INFORMATION MATRICES


# DOWNLOAD THE SAMPLE INFORMATION MATRICES
# Sheets: res_sheet = prediction float values for each class 
# 0 = wt, 1 = MED12, 2 = HMGA2, 3 = YEATS4

save_path_sheet = '/lustre/scratch/kiviaho/myoma/myoma-new/grad-cam-maps/sheets/'
save_path_hmap = '/lustre/scratch/kiviaho/myoma/myoma-new/grad-cam-maps/heatmaps/'

res_sheet = 'multilabel_KF-1-4_1_epochs_100_percent_21875130_at_2022-06-12_19:39:50_598px.h5new_dataset_multiclass_prediction_results.csv'
tfrecord_contents = pd.read_csv('/lustre/scratch/kiviaho/myoma/myoma-new/KF_5_in_5_tfr_contents_in_order.tsv')

results = pd.read_csv('/lustre/scratch/kiviaho/myoma/myoma-new/'+res_sheet)
results.dropna(inplace=True)
results = results.rename(columns={'0': 0, '1': 1,'2': 2,'3': 3})

results = results.loc[:,[0,1,2,3]] # Only keep the float prediction values! All else is false (wrong order)

tfrecord_contents = tfrecord_contents[:len(results)]

# Combine the extracted information & predictions
results = pd.concat([tfrecord_contents.reset_index(drop=True), results], axis=1)
results = results.drop(results.columns[0], axis=1) # Remove old indices



######
type_label = sys.argv[1]
int_label = int(toIntEnum[type_label].value)
n_tiles = int(sys.argv[2])

data = results[results[int_label] == 1][results['Label'] == int_label].reset_index(drop=True)

# Take 1000 random tiles with seed number 100
data = data.sample(n=n_tiles,random_state=100).reset_index(drop=True)
data = get_coordinates(data)
data['hmap_file'] = ''


# Prepare the model and extract a layer
model = keras.models.load_model('/lustre/scratch/kiviaho/myoma/myoma-new/results/multilabel_KF-1-4_1_epochs_100_percent_21875130_at_2022-06-12_19:39:50_598px.h5')
# Remove last layer's activation
model.layers[-1].activation = None
img_size = (598, 598,3)

last_conv_layer_name = "conv2d_93"

# First, we create a model that maps the input image to the activations
# of the last conv layer as well as the output predictions
grad_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
)

hostdevice = '/device:gpu:0'
with tf.device(hostdevice):
    for idx in range(n_tiles):
        img_path = data['Path'][idx]

        heatmap_file_name = str(type_label) \
        + '_' + str(data['Sample'][idx]) \
        + '_' + str(data['x_coord'][idx]) \
        + '_' + str(data['y_coord'][idx]) \
        + '_grad_cam_hmap_values.npy'

        img_array = get_img_array(img_path)

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, grad_model, last_conv_layer_name,pred_index=int_label)
        np.save(save_path_hmap+type_label+'/'+heatmap_file_name,heatmap)

        data['hmap_file'][idx] = save_path_hmap+type_label+'/'+heatmap_file_name

        if idx % 100 == 0:
            data.to_csv(save_path_sheet+str(type_label)+'_'+str(n_tiles)+'_tiles_grad_cam_hmap.tsv',sep='\t')
            print('Saving hmaps ' + str(idx)+ '/' + str(n_tiles))
    data.to_csv(save_path_sheet+str(type_label)+'_'+str(n_tiles)+'_tiles_grad_cam_hmap.tsv',sep='\t')