import os
import sys
import glob
import pickle
import argparse
import time

import numpy as np
import pandas as pd

from functools import partial
from logzero import logger
from matplotlib import pyplot as plt
from enum import IntEnum

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

import tensorflow as  tf
from keras import backend as K

# EXAMPLE RUN: python -u --t test-tfrecords.tsv --o test-tile-level-data.tsv extract-tfr-info.py

# Sample labeling scheme for string to enum
# Import in all scripts necessary
from enum import IntEnum
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
      'sample': tf.io.FixedLenFeature([], tf.string),
      'int_label': tf.io.FixedLenFeature([], tf.int64),
      'path': tf.io.FixedLenFeature([], tf.string),
      'label':  tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    return example


def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.prefetch(buffer_size=50)
    dataset = dataset.map(reshape_data)
    return dataset


def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(
        filenames
        )  # automatically interleaves reads from multiple files
    dataset = dataset.map(
        partial(parse_tfrecord_fn), num_parallel_calls=50
    )
    # returns a dataset of (, label) pairs
    return dataset

def reshape_data(dataset):
    return dataset['int_label'],dataset['sample'],dataset['path'] 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extracting information from tfrecords")
    parser.add_argument('-t', '--tiles', dest='tiles_file', required=True,
                        type=argparse.FileType('r'),
                        help='file containing list of tiles path')
    parser.add_argument('-o', '--outfile', dest='out_file', required=True,
                        help='name of the output file')


    ## ARGS ##
    args = parser.parse_args()
    

    ### DATA IMPORT ###

    path = ''
    sample_sheet = pd.read_csv(args.tiles_file, sep = '\t')
    files = list(dict.fromkeys(sample_sheet['TFRecords'])) # This preserves the order, unlike set !!
    files = [path + f  for f in files]

    # Parsing the dataset
    logger.info('Parsing the dataset')
    test_dataset = get_dataset(files)
    
    
    samples  =[]
    paths = []
    labels = []
    i = 1
    N_SAMPLES = len(sample_sheet)
    for label, sample, path in test_dataset:
        samples.append(sample.numpy().decode())
        paths.append(path.numpy().decode())
        labels.append(label.numpy())
        i+=1
        if i % 100000 == 0:
            logger.info("SAMPLE "+str(i)+"/"+str(N_SAMPLES*1600)) # 1600 = the number of tiles in a tfrecord
            
    tfrecord_data = pd.DataFrame({
        "Sample":samples,
        "Path":paths,
        "Label":labels})
            
    tfrecord_data.to_csv(args.out_file)
        
        
        
        
