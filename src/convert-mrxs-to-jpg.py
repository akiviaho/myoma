import os
import sys
import cv2
import math
import logging
import numpy as np
import imageio

from pathlib import Path
from openslide import open_slide, ImageSlide

print(sys.version)
sys.path.append(os.getcwd())

"""
Example run:

convert-mrxs-to-jpg.py Slides-20220326-myoma-JPG slide_names.txt

slide_names: A file containing tile path seperated by newline.
"""


if __name__ == "__main__":


    # Create necessary directories for saving
    directory = sys.argv[1] 
    directory = Path(directory)     # convert into path type

    if not directory.exists:
        os.mkdir(directory)
    else:
        logging.info("%s Skipping  existing  ", directory)

    # Read the command line input image sheet file
    with open(sys.argv[2]) as f:
        slide_sheet =f.read().splitlines()
    
    n_samples = len(slide_sheet)
    idx = 1

    for slide_path in slide_sheet:
        print("Converting slide {:n}/{:n}".format(idx,n_samples))
        # See: https://openslide.org/api/python/ for more
        dir_name = os.path.dirname(slide_path)       # absolute path of input slide
        file_name = os.path.basename(slide_path).replace(".mrxs", ".jpg")

        full_slide = open_slide(slide_path)     # slide image in full resolution
        
        print('')
        print(file_name)
        print('Level downsamples: '+str(full_slide.level_downsamples))
        print('Level dimensions: '+str(full_slide.level_dimensions))

        downsample_lvl_idx = 4 # Controls the level of downsampling (disk space) 4 == 16x downsample
        rgb_img = full_slide.get_thumbnail(full_slide.level_dimensions[downsample_lvl_idx])   
        imageio.imwrite(directory/file_name, np.asarray(rgb_img),
                        optimize=True, quality=85)
        
        idx +=1
        

                        
