import os
import sys
import argparse

import numpy as np
import pandas as pd

from functools import partial
from logzero import logger
from enum import IntEnum
import tensorflow as  tf
from keras.models import load_model

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

# Sample labeling scheme for string to enum
# Import in all scripts necessary
class toIntEnum(IntEnum):
    MED12 = 0
    HMGA2 = 1
    UNK = 2
    HMGA1 = 3
    OM = 4
    YEATS4 = 5

    
### DEFINITIONS ###
# This is for predicting
def parse_tfrecord_fn(example):
    feature_description = {
      'img': tf.io.FixedLenFeature([], tf.string),
      'sample': tf.io.FixedLenFeature([], tf.string),
      'int_label': tf.io.FixedLenFeature([], tf.int64),
      'path': tf.io.FixedLenFeature([], tf.string),
      'label':  tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['img'], channels=3)/255
    
    # Flip along the horizontal axis and rotate randomly (no biases)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image,k=int(np.random.choice((1,2,3,4),1)))
    example['img'] = image
    
    # Modify labels to binary if necessary
    if N_TARGETS ==2:
        example['int_label'] = tf.cond(tf.math.greater(example['int_label'], 0), lambda: 1, lambda: 0)
    example['int_label'] = tf.one_hot(example['int_label'],N_TARGETS)
    return example


def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.prefetch(buffer_size=50)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)
    dataset = dataset.map(reshape_data)
    return dataset


def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(
        filenames
        )  # automatically interleaves reads from multiple files
    dataset = dataset.map(
        partial(parse_tfrecord_fn), num_parallel_calls=50
    )
    # returns a dataset of (image, label) pairs
    return dataset

def reshape_data(dataset):
    image = tf.reshape(dataset['img'],(BATCH_SIZE,*IMAGE_SIZE,3))
    label = tf.reshape(dataset['int_label'],(BATCH_SIZE,N_TARGETS))
    return image,label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting leiomyoma tiles \
                                     with InceptionV3 model")
    parser.add_argument('-m', '--model', required=True,
                            help='trained model for prediction')
    parser.add_argument('-tfr', '--tfrecords', dest='records_file', required=True,
                        type=argparse.FileType('r'),
                        help='file containing list of tfrecords')

    parser.add_argument('--binary_model', action='store_true',
                   help='Predict with a binary model')
    parser.add_argument('--multiclass_model', dest='binary_model', action='store_false',
                       help='Predict with a multiclass model')
    parser.set_defaults(binary_model=False)
    
    parser.add_argument('-iw', '--img_width', type=int,
                        help='Input image width')
    parser.add_argument('-ih', '--img_height', type=int,
                        help='Input image height')
    parser.add_argument('-b', '--batch', type=int,
                    help='Batch size for generator')


    ## ARGS ##
    args = parser.parse_args()
    BATCH_SIZE = args.batch
    IMAGE_SIZE = (args.img_width,args.img_height)
    if args.binary_model:
        logger.info('Using a binary model...')
        N_TARGETS = 2
    else:
        N_TARGETS = 6
        logger.info('Using a multiclass model...')
    
    ### DATA IMPORT ###


    records = pd.read_csv(args.records_file, sep = '\t')

    path = '' # Fix for problems with path
    files = [path + f  for f in records['TFRecords']]
    

    # Parsing the dataset
    logger.info('Parsing the dataset...')
    test_dataset = get_dataset(files)
    
    logger.info('Loading the model...')
    model_name = args.model
    model = load_model('/lustre/scratch/kiviaho/myoma/myoma-new/results/'+ model_name)

    # PREDICT 
    logger.info('Predicting labels...')
    pred = model.predict(test_dataset)

    # Saving the prediction results
    results_for_saving = pd.DataFrame(pred)
    
    if N_TARGETS == 2:
        name = model_name.strip('.h5') + '_binary_prediction_results.csv'
        results_for_saving.to_csv(name, index=False)
    
    else:
        name = model_name.strip('.h5') + '_multiclass_prediction_results.csv'
        results_for_saving.to_csv(name, index=False)
    
    logger.info('All done!')
    
    