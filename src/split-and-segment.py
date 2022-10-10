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

python src/split-and-segment.py  slide_names.txt

slide_names: A file containing tile path seperated by newline.
"""


def mask_image(mask, image, mask_color=(0, 255, 0,)):
    '''
    mask target region with green color
    '''
    masked = image.copy()
    masked[mask != 0] = mask_color
    return masked


def compute_otsu_mask(image):
    '''
    Calcualte OTSU adaptive threshold
    and return resulting np.array and threshold value
    '''
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out, thresh = cv2.threshold(image_grayscale, 0, 255,
                                cv2.THRESH_BINARY_INV + (cv2.THRESH_OTSU))
    return out, thresh


def get_slice(full_slide, location, level=0, size=(598, 598)):
    '''
    Get target region from whole slie image using
    openslide .read_region function
    '''

    sliced = full_slide.read_region(location, level, size)
    return sliced


if __name__ == "__main__":
    x_tile = 598
    y_tile = 598

    # Create necessary directories for saving
    directory = "tiles_161_180"
    directory = Path(directory)     # convert into path type

    directory_for_mask = "old_slides_otsu_segmentation_masks"
    directory_for_mask = Path(directory_for_mask)

    if not directory.exists:
        os.mkdir(directory)
    else:
        logging.info("%s Skipping  existing  ", directory)

    if not directory_for_mask.exists:
        os.mkdir(directory_for_mask)
    else:
        logging.info("%s Skipping  existing  ", directory_for_mask)

    # Read the command line input image sheet file
    with open(sys.argv[1]) as f:
        slide_sheet =f.read().splitlines()
    
    n_samples = len(slide_sheet)
    idx = 1

    for slide_path in slide_sheet:
        print("Segmenting and splitting slide {:n}/{:n}".format(idx,n_samples))

        slide = ImageSlide(slide_path)      # small slide preview size
        full_slide = open_slide(slide_path)     # slide image in full resolution

        # Height and Width of full resolution slide
        f_height = full_slide.dimensions[1]
        f_width  = full_slide.dimensions[0]

        # Choose 1/64 resolution for thumbnail image! See: https://openslide.org/api/python/ for more
        # Previously was 1/128 which is really low. Doesn't really affect performance
        # downsample_lvl = 6 for 64, 7 for 128
        downsample_lvl_idx = 6
        thumb = full_slide.get_thumbnail(full_slide.level_dimensions[downsample_lvl_idx])   
        thumb_downsample_lvl = full_slide.level_downsamples[downsample_lvl_idx]
        image = np.asarray(thumb)  # convert into numpy array

        dir_name = os.path.dirname(slide_path)       # absolute path of input slide
        file_name = os.path.basename(slide_path).replace(".mrxs", "")

        print("Calcualte OTSU threshold and get image array")
        # Calcualte OTSU threshold and get image array
        thresh_otsu, mask_otsu  = compute_otsu_mask(image)

        # get masked image
        masked_img = mask_image(mask_otsu, image)

        # check if the input is tumor or wildtype sample
        sample_type = ""
        sample      = file_name


        # Height and Width of preview size image
        height = masked_img.shape[0]
        width  = masked_img.shape[1]

        # 2nd batch full image size = (88473 x 211878)  (width, height)
        # Multiply the number of points by a factor of two to get

        n_points_width = 2*f_width/x_tile
        n_points_height = 2*f_height/y_tile

        # Equal distance target points to slice from full resolution image
        fw_points = [i for i in range(0, f_width,  math.floor(f_width/n_points_width))]
        fh_points = [i for i in range(0, f_height, math.floor(f_height/n_points_height))]

        
        # thumbnail size = 1382 x 3310)  (width, height)
        w_points = [math.floor(element/thumb_downsample_lvl) for element in fw_points]
        h_points = [math.floor(element/thumb_downsample_lvl) for element in fh_points]

        thumb_step_x = math.floor(x_tile/thumb_downsample_lvl)
        thumb_step_y = math.floor(y_tile/thumb_downsample_lvl)
        n_pixels_thumb_tile = thumb_step_x * thumb_step_y

        mask_name = file_name + "_" + 'binary_segmentation_mask' + '.jpg'
        imageio.imwrite(directory_for_mask/mask_name, np.asarray(masked_img))

        for indexh, p1 in enumerate(h_points):
            # go left to right
            for indexw, p2 in enumerate(w_points):
                
                # slice region from masked image
                thumb_slice = mask_otsu[p1:p1+thumb_step_y, p2:p2+thumb_step_x]          
            
                # number of pixels masked
                num_of_masked_pixels = (thumb_slice != 0).sum()

                if num_of_masked_pixels >=  n_pixels_thumb_tile*0.95:
                    try:
                        # location to slice
                        location = (fw_points[indexw], fh_points[indexh])

                        # slice region
                        sliced = get_slice(full_slide=full_slide, location=location)
                        # output image name
                        name  = sample_type  + "_" + file_name + "_" + str(location[0]) + '_' + str(location[1]) + '.jpg'
                        
                        # write the image
                        rgb_img = sliced.convert('RGB')
                        imageio.imwrite(directory/name, np.asarray(rgb_img),
                                        optimize=True, quality=85)
                    except:
                        continue

                        
