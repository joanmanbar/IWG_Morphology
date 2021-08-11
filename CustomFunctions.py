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
# Found here: https://github.com/matplotlib/matplotlib/issues/16976/
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
# This function takes an rgb or gray image, or the image's fullpath, and crops it to the area where the stems are clustered.
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
# This function takes an rgb or gray image and changes the color and/or sharpeness values based on Image and ImageEnhance from PILLOW
# Doc: https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html
# Examples: https://medium.com/swlh/image-enhance-recipes-in-pillow-67bf39b63bd
#
#


def EnhanceImage(InputImage, Color = None, Contrast = None, Sharp = None):
    
    # Read image
    if isinstance(InputImage, str) == True:
        img = iio.imread(InputImage)
    else: 
        img = InputImage
    
    # RGB enhancement
    img0 = Image.fromarray(img)
    
    # Color seems to be good around 3.5
    img1 = ImageEnhance.Color(img0)
    if Color is not None:
        img1 = img1.enhance(Color)
    else:
        img1 = img0
    
    # Contrast
    img2 = ImageEnhance.Contrast(img1)
    if Contrast is not None:
        img2 = img2.enhance(Contrast)
    else:
        img2 = img1
    
    # Sharpness (Good ~20 or higher)
    img3 = ImageEnhance.Sharpness(img2)    
    if Sharp is not None:
        img3 = img3.enhance(Sharp)
    else:
        img3 = img2
    
    # Final image
    img3 = np.array(img3)
    
    return img3
    


#-----------------------------------------------------#
#               Compare two plots
#-----------------------------------------------------#
# Taken from scikit-image: 
# Link: https://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html
#
#


def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')







#-----------------------------------------------------#
#               Remove spike's background
#-----------------------------------------------------#
#
# Function to remove background of spikes from 2019
#

def RemoveBackground(img, rgb_out = True, Thres_chan = 0, Thres_val = 50):
    
    # Read image
    if isinstance(img, str) == True:
        img0 = iio.imread(img)
    else: 
        img0 = img
    
    img1 = img0[44:6940, 25:4970, :]
    
    # Threshold based on channel
    bw0 = img1[:, :, Thres_chan] > Thres_val
    
    # Remove small objects
    n_pixels = img1.shape[0] * img1.shape[1]
    minimum_size = n_pixels/10000
    bw1 = morphology.remove_small_objects(bw0, min_size=np.floor(minimum_size))
    
    # Apply mask
    img2 = np.where(bw1[..., None], img1, 0) 
    
    if rgb_out == True:
        return img2
    else:
        # Convert to gray
        gray0 = img2 @ [0.2126, 0.7152, 0.0722]
        return gray0
    
    











#-----------------------------------------------------#
#               Compare Multiple Plots
#-----------------------------------------------------#
# Compare multiple plots at once. 
#

def ComparePlots(rows, cols, images):
    plots = rows * cols
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    ax = axes.ravel()
    for i in range(plots):
        ax[i].imshow(images[i], cmap='gray')
        Title = "Image " + str(i)
        ax[i].set_title(Title, fontsize=20)
    fig.tight_layout()
    plt.show()











