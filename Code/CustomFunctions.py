#!/usr/bin/env python3


#-----------------------------------------------------#
#               CUSTOM FUNCTIONS
#-----------------------------------------------------#
#
# This .py file contains the definitions to some useful custom functions
#
#
#


















# Dependencies

import time
# start_time = time.time()

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

from PIL import Image, ImageEnhance 

# import scipy
# from scipy import ndimage as ndi

# from PIL import Image, ImageEnhance 

# import cv2 as cv

# import math

# import pandas as pd                          

# Random label cmap
import matplotlib
import colorsys






#-----------------------------------------------------#
#               Random Label cmap
#-----------------------------------------------------#
#
#   Found here: https://github.com/matplotlib/matplotlib/issues/16976/
# Used for StarDist
#
#

def random_label_cmap(n=2**16):

    h,l,s = np.random.uniform(0,1,n), 0.4 + np.random.uniform(0,0.6,n), 0.2 + np.random.uniform(0,0.8,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

lbl_cmap = random_label_cmap()






#-----------------------------------------------------#
#               Cropping Function
#-----------------------------------------------------#
#   This function takes an rgb or gray image, or the image's fullpath, and crops it to the area where the stems are clustered.
#
#
# # For debugging
# # Images folder (change extension if needed)
# Images = io.ImageCollection(r'..\Images\Stems\*.JPG')
# rgb = Images.files[0]


def CropStems(InputImage, rgb_out = False, sigma = 0.9, low_threshold = 0, high_threshold = .75):
    
    # Read image
    if isinstance(InputImage, str) == True:
        rgb = iio.imread(InputImage)
        gray0 = rgb @ [0.2126, 0.7152, 0.0722]
    elif len(InputImage.shape) == 3:
        rgb = InputImage
        gray0 = rgb @ [0.2126, 0.7152, 0.0722]
    else: 
        gray0 = InputImage
    
    # Normalize
    gray0 = gray0/255
    
    # Detect edges
    edges = feature.canny(gray0, sigma = sigma, low_threshold = low_threshold, high_threshold = high_threshold)
    # plt.imshow(edges, cmap = 'gray')
    
    # Dilate
    dilated = binary_dilation(edges, selem=morphology.diamond(10), out=None)
    # plt.imshow(dilated, cmap = 'gray')
    
    # Get convex hull
    chull = convex_hull_object(dilated, connectivity=2)
    # plt.imshow(chull)
    
    cropped = np.asarray(chull)
    
    if rgb_out == False:
        cropped = np.where(chull, gray0, 0)
        cropped = cropped*255
    else:
        cropped = np.where(chull[..., None], rgb, 0)

    cropped = cropped.astype(np.uint8)
    # Crop image
    [rows, columns] = np.where(chull)
    row1 = min(rows)
    row2 = max(rows)
    col1 = min(columns)
    col2 = max(columns)
    cropped = cropped[row1:row2, col1:col2]
    
    # plt.imshow(cropped)
    
    return cropped


#-----------------------------------------------------#
#               Enhance Image
#-----------------------------------------------------#
#   This function takes an rgb or gray image and changes the color or sharpeness values based on Image and ImageEnhance from PIL
#
#



def EnhanceImage(InputImage, Color = 0, Sharp = 0):
    
    # Read image
    if isinstance(InputImage, str) == True:
        img = iio.imread(InputImage)
    else: 
        img = InputImage
    
    # RGB enhancement
    img0 = Image.fromarray(img)
    img1 = ImageEnhance.Color(img0)
    
    if len(str(Color)) > 1:
        img1 = img1.enhance(Color)
    else:
        img1 = img1.enhance(3.5)
    
    # Sharpness (Good ~20 or higher)
    img2 = ImageEnhance.Sharpness(img1)
    
    if len(str(Sharp)) > 1:
        img2 = img2.enhance(Sharp)
    else:
        img2 = img2.enhance(20)
    
    # Final image
    img2 = np.asarray(img2)
    
    return img2
    
