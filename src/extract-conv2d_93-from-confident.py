# Date: 26.7.2022
# Author: Antti Kiviaho
#
# Extract conv2d_93 layer from confidently and correctly predicted tiles


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import argparse
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extracting a convolutional layer from confidently predicted tiles.")
    
    # FILES
    parser.add_argument('-pf', '--predictions_file', dest='predictions_file', required=True,
                        type=str, help='file containing the prediction results from testing')
    
    parser.add_argument('-mf', '--model_file', dest='model_file', required=True,
                    type=str, help='file containing the trained model')
    
    parser.add_argument('-if', '--sample_info_file', dest='sample_info_file', required=True,
                type=str, help='file containing tile information extracted from tfrecords')
    
    parser.add_argument('-smf', '--save_metadata_file',dest='save_metadata_file', required=True,
                type=str,help='path to save metadata  to')
    
    parser.add_argument('-scf', '--save_conv_layer_file',dest='save_conv_layer_file', required=True,
            type=str,help='path to save convolutional files  to')
    
    
    

    ################ PATHS & FILES ################
    args = parser.parse_args()
    
    save_metadata_file = args.save_metadata_file
    save_conv_layer_file = args.save_conv_layer_file
    
    
    predictions_file = args.predictions_file
    model_file = args.model_file

    sample_info_file = args.sample_info_file



    ################ FUNCTION DEFINITIONS ################
    def get_img_array(img_path, size=(598,598,3)):
        # `img` is a PIL image of size 598 x 598 x 3
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        # `array` is a float32 Numpy array of shape (299, 299, 3)
        array = keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 598, 598, 3)
        array = np.expand_dims(array, axis=0)
        array = np.divide(array,255,dtype=np.float16)
        return array

    ############# LOAD DATA #############

    tfrecord_contents = pd.read_csv('/lustre/scratch/kiviaho/myoma/myoma-new/'+sample_info_file)

    predictions = pd.read_csv('/lustre/scratch/kiviaho/myoma/myoma-new/'+predictions_file)
    predictions = predictions.rename(columns={'0': 0, '1': 1,'2': 2,'3': 3,'4':4,'5':5})

    # Combine the extracted information & predictions
    results = pd.concat([tfrecord_contents.reset_index(drop=True), predictions], axis=1)
    results = results.drop(results.columns[0], axis=1) # Remove old indices
    results.dropna(inplace=True) # Remove rows with missing values

    # Find tiles with confident prediction that matches ground truth
    conf_pred = results.iloc[np.where(results.loc[:,[0,1,2,3,4,5]]==1.0)[0]]
    conf_pred['Prediction'] = conf_pred.loc[:,[0,1,2,3,4,5]].idxmax(axis=1)
    conf_pred = conf_pred[conf_pred['Label'] == conf_pred['Prediction']].reset_index(drop=True)
    
 
    print(str(len(conf_pred))+ ' tiles confidently predicted in this fold.')

    ############# LOAD MODEL #############

    model = keras.models.load_model('/lustre/scratch/kiviaho/myoma/myoma-new/results/'+model_file)
    # Remove last layer's activation
    model.layers[-1].activation = None
    img_size = (598, 598,3)

    last_conv_layer_name = 'conv2d_93'

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    ############# SAVE METADATA PANDAS AND THE LAYERS AS .npy #############
    conf_pred.to_csv(save_metadata_file, sep='\t',index = False)
    
    
    ############# EXTRACT con2d_93 LAYERS #############
    outputs = []
    
    i = 1
    hostdevice = '/device:gpu:0'
    with tf.device(hostdevice):
        for path in conf_pred['Path']:
            img = get_img_array(path)
            output = grad_model(img)
            outputs.append(np.squeeze(output[0]).flatten())
            if i % 10000 ==0:
                print('Tile '+ str(i) + '/' + str(len(conf_pred)))

                ############# SAVE THE LAYERS AS .npy #############
                np.save(save_conv_layer_file,np.asarray(outputs))

            i+=1
        outputs = np.asarray(outputs)
        
    ############# FINAL TIME SAVING THE LAYERS AS .npy #############
    np.save(save_conv_layer_file,outputs)



