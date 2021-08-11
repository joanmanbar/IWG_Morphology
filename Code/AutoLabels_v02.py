#!/usr/bin/env python3


#########################################################
#
#           Attempt to generate auto labels
#
#########################################################



# Dependencies
from CustomFunctions import *

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
from skimage.filters import try_all_threshold
import skimage

from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image, skeletonize_3d, medial_axis, thin  # noqa
from skimage.morphology import disk, square  # noqa


img0 = r"J:\My Drive\PROJECTS\IWG_Morphology\QP_Project\labeled\0016.tif"
img0= r"J:\My Drive\PROJECTS\IWG_Morphology\QP_Project\export\0016-labels.png"
img1=r"J:\My Drive\PROJECTS\IWG_Morphology\Images\Stems\Train\masks\_MG_0024.JPG.tif"
img0 = iio.imread(img0)
img1 = iio.imread(img1)
img2 = np.uint8(img1)
plt.imshow(img0)
plt.imshow(img1)

ComparePlots(2,2,[img0[:,:,0], img0[:,:,1], img0[:,:,2], img0[:,:,3]])

Images = io.ImageCollection(r'..\Images\Stems\Original\*.JPG')

# Read image
rgb_name = Images.files[3]
rgb = iio.imread(rgb_name)
# plt.imshow(rgb)

labelled = "J:\My Drive\PROJECTS\IWG_Morphology\QP_Project\labeled\0016.tif"
labelled = iio.imread(labelled)
testing = labelled[:,:,0]
plt.imshow(testing)
plt.imshow(labelled)

real_label = iio.imread("J:\My Drive\PROJECTS\IWG_Morphology\Images\Stems\Train\masks\_MG_0024.JPG.tif")
plt.imshow(real_label)


#-----------------------------------------------------#
#               Detect Stems and Enhance Image
#-----------------------------------------------------#


# Crop to rgb stems
img = CropStems(rgb, rgb_out = True)
# plt.imshow(img)
# Or gray
# gray0 = CropStems(rgb, rgb_out = False)

# Enhance image
enh0 = EnhanceImage(img, Color = None, Contrast = -0.8, Sharp = None)
plt.imshow(enh0, cmap='gray')
# I got great results with Contrast ~= -0.8                (Joan - May5,2021)



edge_sobel = filters.sobel(enh0)
edge_scharr = filters.scharr(enh0)
edge_prewitt = filters.prewitt(enh0)

ComparePlots(1,3, [edge_sobel, edge_scharr, edge_prewitt])


frangi = skimage.filters.frangi(enh0, black_ridges=False)
plt.imshow(frangi)


blurry = filters.gaussian(edge_sobel, sigma = 5)
plt.imshow(blurry)







# Skeletonize
sk = skeletonize(gray1 > 20)
plt.imshow(sk, cmap='gray')




# enh0_n = enh0/enh0.max()
# plt.imshow(enh0_n, cmap='gray')

# testing = enh0 @ [1, 1, 0]
# plt.imshow(testing, cmap='gray')

# bw0 = enh0[:, :, 2] > 10
# plt.imshow(bw0, cmap='gray')



enh1 = EnhanceImage(enh0, Color = 3, Contrast = None, Sharp = None)
plt.imshow(enh1, cmap='gray')



#-----------------------------------------------------#
#               Morphological Filtering
#-----------------------------------------------------#


gray0 = enh0 @ [0.2126, 0.7152, 0.0722]
# plt.imshow(gray0, cmap='gray')

# Invert scale. Add 1 so there is no Inf
gray1 = gray0.max()/ (gray0+1)
plt.imshow(gray1, cmap='gray')

# Skeletonize
sk = skeletonize(gray1 > 20)
plt.imshow(sk, cmap='gray')

# Skeletonize (medial axis)
skma, distance = medial_axis(gray1 > 20, return_distance=True)
plt.imshow(skma, cmap='gray')

plot_comparison(sk, skma, "Title")

# Thin
thin_sk = thin(sk, max_iter=50)
thin_skma = thin(skma, max_iter=50)

plot_comparison(thin_sk, thin_skma, "Title")


sk = skeletonize_3d(enh1)
plt.imshow(sk, cmap='gray')


sk_3d = skeletonize(enh0)



# # IDEA:
#     Fill up holes. Some of them will be stems (more circular), others will not. Skeletonize the image and remove those skeletons that are to large, which may be keep only stems (small skeletons)
#   Look at the blobs example: https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html















gray2 = 255/(gray0+1)
plot_comparison(gray1, gray2, "Comparison")

enh2 = EnhanceImage(gray1, Color = None, Contrast = None, Sharp = 5)
plt.imshow(enh0, cmap='gray')


normalized_gray = gray0/gray0.max()
plt.imshow(1/ (normalized_gray ** 2), cmap = 'gray')

plt.imshow(gray0/gray0.max()) 






selem = disk(10)
eroded = white_tophat(gray1, selem)
plot_comparison(gray1, eroded, 'erosion')

sk = skeletonize(gray1 > 20)
sk2 = skeletonize(eroded > 20)
plot_comparison(sk, sk2, 'skeletonize')



inverted = gray0.max()/ (gray0+1)

fig, ax = try_all_threshold(inverted, figsize=(10, 8), verbose=False)
plt.show()


















#-----------------------------------------------------#
#               Watershed
#-----------------------------------------------------#











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






