#!/usr/bin/env python3


#########################################################
#
#           Attempt to generate auto labels
#
#########################################################



# Dependencies
from CustomFunctions import CropStems, EnhanceImage

import time
# start_time = time.time()

import os

import numpy as np

import matplotlib.pyplot as plt

from skimage import io, feature, filters, color, util, morphology, exposure, segmentation, img_as_float
from skimage.filters import unsharp_mask
from skimage.measure import label, regionprops, perimeter, find_contours
from skimage.morphology import medial_axis, skeletonize, convex_hull_image, binary_dilation, black_tophat, diameter_closing, area_opening, erosion, dilation, opening, closing, white_tophat, reconstruction, convex_hull_object, binary_erosion
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from skimage.util import invert

from glob import glob

import imageio as iio

import scipy
from scipy import ndimage as ndi

from PIL import Image, ImageEnhance 

import cv2

import math

import pandas as pd                          





import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.util import img_as_ubyte




Images = io.ImageCollection(r'..\Images\Stems\*.JPG')

#-----------------------------------------------------#
#               Hough Transform
#-----------------------------------------------------#

# Read image
rgb_name = Images.files[0]
rgb = iio.imread(rgb_name)
# plt.imshow(rgb)

# Crop to gray stems
img = CropStems(rgb, rgb_out = True)
plt.imshow(img)

gray0 = CropStems(rgb, rgb_out = False)

enh0 = EnhanceImage(rgb)




#-----------------------------------------------------#
#               Hough Transform
#-----------------------------------------------------#


def hough_circle(img, min_dist, max_radius):
    output = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img2 = np.floor(img)
    # img2 = img2.astype(int)
    gray = cv.medianBlur(gray, 5)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, min_dist,
                              param1=50, param2=30, minRadius=0, maxRadius=max_radius)
    detected_circles = np.uint16(np.around(circles)) # its a list of circle parameters (x, y ,radius)
    for (x, y ,r) in detected_circles[0, :]:
        cv.circle(output, (x, y), r, (255, 0, 0), -1)
        # cv.circle(output, (x, y), 0, (0, 255, 0), 3)
        
    return output, detected_circles # output is the orig image with cirlcles drawn on it


test_hc, test_dc = hough_circle(img, 70, 40)

red = test_hc[:, :, 0]
# green = image1[:, :, 1]
# blue = image1[:, :, 2]

# Threshold based on the red channel (this depends on the image's background)
bw0 = test_hc[:, :, 0] == 255
plt.imshow(bw0)

labeled_stems, num_stem = label(bw0, return_num = True)
plt.imshow(labeled_stems, cmap = lbl_cmap)


# Dilate
eroded = opening(bw0, out=None)
# plt.imshow(eroded, cmap = 'gray')



distance = ndi.distance_transform_edt(bw0)
   # io.imshow(distance)
   # local_maxi = feature.peak_local_max(distance, indices=False, footprint=morphology.diamond(30), labels=myspk_rot)
local_maxi = feature.peak_local_max(distance, indices=False, min_distance=10, labels=bw0)
# stem = morphology.remove_small_objects(local_maxi, min_size=5)
# io.imshow(img_as_float(local_maxi) - img_as_float(stem))
   
# new_local_max = img_as_float(local_maxi) - img_as_float(stem)
# new_local_max = new_local_max.astype(np.bool)
   
 # local_maxi = feature.corner_peaks(distance, indices=False, min_distance=20, labels=myspk_rot)
   # io.imshow(new_local_max)
   
   
   
markers = ndi.label(local_maxi)[0]
labeled_spikelets = segmentation.watershed(-distance, markers, mask=bw0)
plt.imshow(labeled_spikelets)

regions_spikelets = regionprops(labeled_spikelets)

# n_Spikelets = int(labeled_spikelets[:,:].max())

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(bw0, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labeled_spikelets, cmap=lbl_cmap)
ax[2].set_title('Separated objects')

fig.tight_layout()
plt.show()






