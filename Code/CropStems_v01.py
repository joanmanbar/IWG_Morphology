#!/usr/bin/env python3



# Crop images to stem area



# Dependencies

import time
start_time = time.time()

import os

import numpy as np

import matplotlib.pyplot as plt

from skimage import io, feature, filters, color, util, morphology, exposure, segmentation, img_as_float
from skimage.filters import unsharp_mask
from skimage.measure import label, regionprops, perimeter, find_contours
from skimage.morphology import medial_axis, skeletonize, convex_hull_image, binary_dilation, black_tophat, diameter_closing, area_opening, erosion, dilation, opening, closing, white_tophat, reconstruction, convex_hull_object
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from skimage.util import invert

from glob import glob

import imageio as iio

import scipy
from scipy import ndimage as ndi

from PIL import Image, ImageEnhance 

import cv2 as cv

import math

import pandas as pd                          

# Random label cmap
import matplotlib
import colorsys
def random_label_cmap(n=2**16):

    h,l,s = np.random.uniform(0,1,n), 0.4 + np.random.uniform(0,0.6,n), 0.2 + np.random.uniform(0,0.8,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

lbl_cmap = random_label_cmap()






# ## For debugging
# # Images folder (change extension if needed)
# Images = io.ImageCollection(r'..\Images\Stems\*.JPG')
# rgb = Images.files[0]

#-----------------------------------------------------#
#               Cropping Function
#-----------------------------------------------------#

def CropStems(rgb):
    
    # Read image
    rgb_name = rgb
    rgb = iio.imread(rgb_name)
    # plt.imshow(rgb)
    
    
    # rgb to gray
    gray0 = rgb @ [0.2126, 0.7152, 0.0722]
    
    # Normalize
    gray0 = gray0/255
    
    # Detect edges
    edges = feature.canny(gray0, sigma = 0.9, low_threshold = 0, high_threshold = .75)
    # plt.imshow(edges, cmap = 'gray')
    
    # Dilate
    dilated = binary_dilation(edges, selem=morphology.diamond(10), out=None)
    # plt.imshow(dilated, cmap = 'gray')
    
    # Get convex hull
    chull = convex_hull_object(dilated, connectivity=2)
    # plt.imshow(chull)
    
    # Apply mask to gray
    gray1 = np.asarray(chull)
    gray1 = np.where(chull, gray0, 0)
    # Or, apply mask to rgb
    # new_rgb = np.where(chull[..., None], rgb, 0)
    
    # Crop image
    [rows, columns] = np.where(chull)
    row1 = min(rows)
    row2 = max(rows)
    col1 = min(columns)
    col2 = max(columns)
    cropped = gray1[row1:row2, col1:col2]
    # plt.imshow(cropped)
    
    
    # Verify folder exists
    if os.path.isdir('..\Images\Stems\CroppedStems') == False:
        os.mkdir('..\Images\Stems\CroppedStems')
    
    # image name
    # cropped_name = "cropped_" + rgb_name
    cropped_name = rgb_name.split("\\")[-1]
    cropped_name = cropped_name.split(".")[-2]
    cropped_name = "..\Images\Stems\CroppedStems\\" + cropped_name + "_cropped.JPG"
    
    # Save image
    im = cropped*255
    im = Image.fromarray(im)
    im = im.convert("L")
    im.save(cropped_name)
    
    # return gray1
    








#-----------------------------------------------------#
#               Executing the function
#-----------------------------------------------------#
#             VERIFY FOLDER AND EXTENSION!!!
#-----------------------------------------------------#

# Images folder (change extension if needed)
Images = io.ImageCollection(r'..\Images\Stems\*.JPG')

# Loop through images in folder
for i in Images.files:
    
    # Set the initial time per image
    image_time = time.time()
    
    # Read image
    rgb = i
    
    # Crop image and convert to gray
    CropStems(rgb)
    
    # How long did it take to run this image?
    print("The image", i.split('\\')[-1], "took", time.time() - image_time, "seconds to run.")
    



# How long did it take to run the whole code?
print("This entire code took", time.time() - start_time, "seconds to run.")

    


