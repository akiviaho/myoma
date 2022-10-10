# Author: Antti Kiviaho
# Date: 16.5.2022 
#
# New workflow for mutation type classification


import os
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from logzero import logger
from pathlib import Path
from enum import IntEnum

import tensorflow as tf

## Speeds up training by ~50%
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

from keras import Model
from keras import utils
from keras import backend as K
from keras.layers import Dense

from tensorflow.keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from functools import partial


# Fix for saving trained model into disk
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
logger.info("Python module imports completed.")

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
def parse_tfrecord_fn(example):
    feature_description = {
      'img': tf.io.FixedLenFeature([], tf.string),
      'int_label': tf.io.FixedLenFeature([], tf.int64),
      'label':  tf.io.FixedLenFeature([], tf.string),
    }
    # IMAGE DATA MUTATING FUNCTION HERE:
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['img'], channels=3)/255
    
    # Flip along the horizontal axis and rotate randomly (no biases)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image,k=int(np.random.choice((1,2,3,4),1)))
    example['img'] = image
    # Mofifies to match two labels
    if N_TARGETS == 2:
        example['int_label'] = tf.cond(tf.math.greater(example['int_label'], 0), lambda: 1, lambda: 0)
    example['int_label'] = tf.one_hot(example['int_label'],N_TARGETS)
    
    return example


def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.prefetch(buffer_size=50)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.repeat(np.ceil(EPOCHS*PERCENT/100)) # Makes sure there is enough data for the epochs
    dataset = dataset.map(reshape_data)
    return dataset


def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(
        filenames
        )  # automatically interleaves reads from multiple files
    dataset = dataset.map(
        partial(parse_tfrecord_fn), num_parallel_calls=30
    )
    # returns a dataset of (image, label) pairs
    return dataset  

def reshape_data(dataset):
    image = tf.reshape(dataset['img'],(BATCH_SIZE,*IMAGE_SIZE,3))
    label = tf.reshape(dataset['int_label'],(BATCH_SIZE,N_TARGETS))
    return image,label

def calculate_class_weights(df):
    weights = []
    cols = np.unique(df['Label'])
    for s in cols:
        weights.append(len(df)/np.where(df['Label'] == s)[0].shape[0])
    weights_dict = dict()
    for i in range(len(weights)):
        weights_dict[cols[i]] = np.round(weights[i]/sum(weights),3)
    return weights_dict

def make_multiclass_InceptionV3_model(image_size, n_targets,lr_schedule):
    
    hostdevice = '/device:gpu:0'
    with tf.device(hostdevice):
        base_model = tf.keras.applications.InceptionV3(
            input_shape=(*IMAGE_SIZE, 3), 
            include_top=False, 
            weights=None,
            pooling='avg'
        )
        outputs = Dense(n_targets, activation='softmax')(base_model.output) # Replaced sigmoid with softmax for multiclass
        model = Model(inputs=base_model.input, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="categorical_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc"),'accuracy'],
        )

    return model

def make_binary_InceptionV3_model(image_size, n_targets,lr_schedule):
    
    hostdevice = '/device:gpu:0'
    with tf.device(hostdevice):
        base_model = tf.keras.applications.InceptionV3(
            input_shape=(*IMAGE_SIZE, 3), 
            include_top=False, 
            weights=None,
            pooling='avg'
        )
        outputs = Dense(n_targets, activation='sigmoid')(base_model.output)
        model = Model(inputs=base_model.input, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc"),'accuracy'],
        )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training leiomyoma tiles \
                                     with InceptionV3 model")
    
    # FILES
    parser.add_argument('-tfr', '--tfrecords', dest='records_file', required=True,
                        type=str,
                        help='file containing list of tfrecords')
    parser.add_argument('-ss', '--sample_sheet', dest='sample_sheet', required=True,
                    type=str,
                    help='file containing extracted tfrecord information')
    parser.add_argument('-vt', '--val_tiles', dest='val_tiles_file', required=False,
                type=str,
                help='file containing list of validation set tiles path', )
    
    # MODEL CONFIGURATION
    parser.add_argument('--validation', action='store_true',
                   help='Do validation (requires a validation file)')
    parser.add_argument('--no_validation', dest='validation', action='store_false',
                       help='Skip validation')
    parser.set_defaults(validation=False)
    
    parser.add_argument('--binary_model', action='store_true',
                       help='Train a binary model')
    parser.add_argument('--multiclass_model', dest='binary_model', action='store_false',
                       help='Train a multiclass model')
    parser.set_defaults(binary_model=False)
    
    parser.add_argument('-e', '--epochs', type=int,
                    help='Number of epochs to use in training')
    parser.add_argument('-b', '--batch', type=int,
                        help='Batch size for reading records')
    parser.add_argument('-r', '--learning_rate', type=float,
                        help='learning rate', default=0.001)
    parser.add_argument('-p', '--percent', type=int,
                help='Percentage of epochs to run in training')

    # PARAMETERS
    parser.add_argument('-sd', '--seed_number', type=int,
                    help='Seed number used in shuffling',default=42)
    parser.add_argument('-pre', '--prefix', type=str,
                    help='Prefix to use as when saving the model')
    parser.add_argument('-iw', '--img_width', type=int,
                        help='Input image width')
    parser.add_argument('-ih', '--img_height', type=int,
                        help='Input image height')
    parser.add_argument('-stf', '--size_tfrecord', type=int,
                    help='Number of individual samples in a tfrecord', default = 1600)
    parser.add_argument('-j', '--jobid', help='JOBID from slurm')

    args = parser.parse_args()

    # ARGS FOR CLARITY
    JOBID = args.jobid
    BATCH_SIZE = args.batch
    IMAGE_SIZE = [args.img_width,args.img_height]
    TILES_IN_RECORD = args.size_tfrecord
    EPOCHS = args.epochs
    LR = args.learning_rate
    PERCENT = args.percent
    SEED = args.seed_number
    PREFIX = args.prefix
    
    
    ######################################### IMPORT DATA #########################################
    logger.info('Importing data...')
    
    records = pd.read_csv(args.records_file, sep = '\t')
    path = '' # Fix for problems with path
    files = [path + f  for f in records['TFRecords']]
    
    random.seed(SEED)
    random.shuffle(files)
    
    # Calculate class weights
    sample_sheet = pd.read_csv(args.sample_sheet, sep = ',')
#    labels = [ toIntEnum[s].value for s in sample_sheet['Type'] ]
#    sample_sheet['Label'] = labels
    
    if args.binary_model:
        logger.info('Modifying labeles for binary model...')
        N_TARGETS = 2
        sample_sheet.replace({'Label' : { 2 : 1, 3 : 1, 4 : 1, 5 : 1}},inplace=True)
        training_weights = calculate_class_weights(sample_sheet)

    else:
        N_TARGETS = len(np.unique(sample_sheet['Label']))
        training_weights = calculate_class_weights(sample_sheet)
        
    # SPLIT THE FILES INTO TRAINING AND VALIDATION
    if args.validation:
        logger.info('Validation switch on. Importing validation data...')
        
        training_files = files
        try:
            valid_records = pd.read_csv(args.val_tiles_file, sep = '\t')
            valid_files = [path + f  for f in valid_records['TFRecords']]
            random.shuffle(valid_files)
            valid_dataset = get_dataset(valid_files)           
            # READ IN DATASETS FROM TFRECORD FILENAMES
            train_dataset = get_dataset(training_files)
            
        except:
            sys.exit('No validation file present!')

    else:
        logger.info('Training weights: ' + str(training_weights))
        logger.info('Training files: ' + str(len(files)))
        training_files = files
        train_dataset = get_dataset(training_files)
        
        
    ######################################### CONFIGURE PARAMETERS #########################################
        
    # CALCULATE THE NUMBER OF WHOLE TRAINING BATCHES AVAILABLE
    TRAIN_BATCHES = int(np.floor((len(training_files)*TILES_IN_RECORD)/BATCH_SIZE))
        
    # DEFINE LEARNING RATE DECAY
    learning_schedule = tf.keras.optimizers.schedules.PolynomialDecay(LR,EPOCHS)
    
    stamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_')
    checkpoint_path = 'results/CHECKPOINT_multilabel_KF-1-4_' + str(EPOCHS) + \
             '_epochs_' + str(PERCENT) + '_percent_' + \
             str(JOBID) + '_at_' + stamp + '_598px.h5'

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
             checkpoint_path , save_best_only=True
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
    
    
    ######################################### COMPILE AND TRAIN MODEL #########################################
    
    if args.binary_model:
        logger.info('Compiling binary model...')
        model = make_binary_InceptionV3_model(IMAGE_SIZE, N_TARGETS, learning_schedule)
    
    else: 
        logger.info('Compiling multiclass model...')
        model = make_multiclass_InceptionV3_model(IMAGE_SIZE, N_TARGETS, learning_schedule)
    
    logger.info('Training the model with ' + str(EPOCHS) + ' epochs')
    
    if args.validation:
        model.fit(train_dataset,
              validation_data= valid_dataset,
              epochs=EPOCHS,
              class_weight = training_weights,
              steps_per_epoch=int(TRAIN_BATCHES * (PERCENT/100)))
    else: 
        model.fit(train_dataset,
              epochs=EPOCHS,
              class_weight = training_weights,
              steps_per_epoch=int(TRAIN_BATCHES * (PERCENT/100)))
    
    ######################################### SAVE MODEL #########################################
                    
    logger.info("saving trained model to disk...")
    stamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_')
    Path("results").mkdir(parents=True, exist_ok=True)
                    
    model_name = PREFIX + str(EPOCHS) + \
        '_epochs_' + str(PERCENT) + '_percent_' + \
        str(JOBID) + '_at_' + stamp + '_598px.h5'

    if args.binary_model:
        output = 'results/binary_' + model_name
        logger.info("saving\t %s", output)
        model.save(output)
    
    else: 
        output = 'results/multiclass_' + model_name
        logger.info("saving\t %s", output)
        model.save(output)

    
   
