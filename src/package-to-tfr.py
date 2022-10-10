# Author: Antti Kiviaho
# Date: 13.5.2022
#
# A script for packaging tiles from the new dataset with MED12, HMGA2 and YEATS4
# mutations starting with 20 samples to test an binary model
#
#
# PACKAGING TAKES A STANDARD 200s per 10 tfrecords (16 000 tiles).

import time
import os
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from logzero import logger

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

### Define TFRecords datatypes ###

# Define the enumeration
from enum import IntEnum
class toIntEnum(IntEnum):
    MED12 = 0
    HMGA2 = 1
    UNK = 2
    HMGA1 = 3
    OM = 4
    YEATS4 = 5


# Create datatypes with tf.train.feature
def _image_feature(value):
    """Returns a bytes_list from a array."""
    return tf.train.Feature(
    bytes_list=tf.train.BytesList(value=[value])
    )
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(toIntEnum[value])]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
#     if isinstance(value, type(tf.constant(0))):
#         value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def create_example(image, label, sample, path):
    feature = {
      'img': _image_feature(image),
      'int_label': _int64_feature(label),
      'sample': _bytes_feature(sample),        
      'path': _bytes_feature(path),
      'label':  _bytes_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))



def write_tfrecords_from_csv(sample_set, batch_size, write_path, record_prefix):
    ''' 
    Writes jpgs into a set of tfrecord-files defined by the first argument. 
    Number of jpg's written into a single TFRecord is defined by batch_size.
    
    params:
        sample_set: A data-frame of shape nx3, with columns 'Sample', 'Type' and 'Tile'.
        batch_size: The number of jpg-files to be written into a single TFRecord
        write_path: Write path, relative or absolute
        
    returns:
        sample_set: A dataframe of shape nx4, with column 'TFRecords' specifying the file
        into which the particular tile has been written into.
    '''
    # Define the number of TFRecord-files to be written
    n_batches = np.floor(len(sample_set)/batch_size).astype(int)
    batch_res = len(sample_set)-batch_size*n_batches
    tfrecords = []
    
    # Write the full-batch TFRecords
    for it in range(n_batches):
        if it%10 == 0:
            logger.info("Writing TFRecord "+ str(it) +" of "+ str(n_batches+1))
        start = time.time()
        subset = sample_set[int(batch_size*it):int(batch_size*(it+1))]
        subset = subset.reset_index(drop=True)
        filename = write_path + record_prefix + str(it) + '.tfr'
        tfrecords = np.hstack((tfrecords,np.repeat(filename,batch_size)))
        
        samples = subset['Sample']
        labels = subset['Type']
        img_paths = subset['Tile']
        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(batch_size):
                image = tf.io.read_file(img_paths[i]).numpy()
                example = create_example(image,labels[i], samples[i], img_paths[i])
                writer.write(example.SerializeToString())
    
    # Write the last "partial" batch
    subset = sample_set[int(batch_size*n_batches):int(batch_size*n_batches+batch_res)]
    subset = subset.reset_index(drop=True)
    filename = write_path + record_prefix + str(n_batches+1) + '.tfr'
    tfrecords = np.hstack((tfrecords,np.repeat(filename,batch_res)))

    samples = subset['Sample']
    labels = subset['Type']
    img_paths = subset['Tile']
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(batch_res):
            image = tf.io.read_file(img_paths[i]).numpy()
            example = create_example(image,labels[i], samples[i], img_paths[i])
            writer.write(example.SerializeToString())

    
    sample_set['TFRecords'] = tfrecords.tolist()
    return sample_set


# Parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splitting and writing the data into \
    train and test datasets and writing into TFRecords")
    parser.add_argument('-t', '--tiles', dest='tiles_file', required=True,
                        type=argparse.FileType('r'),
                        help='Tab separated values containing columns Sample, Type & Tile')
    parser.add_argument('-b', '--batch', type=int,
                    help='Batch size to be stored in a single TFRecords file', default=1600)
    parser.add_argument('-sp', '--save_path', type=str,
            help='Path to save records in.',required=True)
    parser.add_argument('-id', '--identifier', type=str,
                help='The identifier to be used in tfrecord name',required=True)
    parser.add_argument('-sd', '--seed_number', type=int,
                        help='Seed number used in shuffling',default=42)
    parser.add_argument('-j', '--jobid', help='JOBID from slurm')
    

    args = parser.parse_args() # Then access by args.member
    
    JOBID = args.jobid

    # Shuffle samples for division into train and test sets
    sample_sheet = pd.read_csv(args.tiles_file,sep="\t")

    # Shuffle samples
    sample_sheet = sample_sheet.sample(frac=1,random_state= args.seed_number).reset_index(drop=True) ## SHUFFLE THE TILE ORDER

    # Change
    save_path = args.save_path
    # Create folders for tfrecords
    Path(save_path).mkdir(parents=True,exist_ok=True)
   
    
    ## WRITE THE TFRECORD-FILES ##
    logger.info("Writing the fold to"+save_path+"...")
    sample_sheet = write_tfrecords_from_csv(sample_sheet, args.batch, save_path,args.identifier)
    
                
    logger.info("Finishing...")

    
    
    
    