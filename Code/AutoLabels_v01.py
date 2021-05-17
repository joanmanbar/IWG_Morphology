# -*- coding: utf-8 -*-

# Atempt to create automatic labels



import numpy as np

import matplotlib.pyplot as plt

from skimage import io, feature, filters, color, util, morphology, exposure, segmentation, img_as_float
from skimage.filters import unsharp_mask
from skimage.measure import label, regionprops, perimeter, find_contours
from skimage.morphology import medial_axis, skeletonize, convex_hull_image, binary_dilation, black_tophat, diameter_closing
from skimage.transform import rescale, resize, downscale_local_mean, rotate

from glob import glob

import imageio as iio

import scipy
from scipy import ndimage as ndi

from PIL import Image, ImageEnhance 




# Gather the sample image files 
Images = io.ImageCollection(r'.\Sample_Images\*.JPG')

# Read and visualize an image
rgb0 = Images[0]
# plt.imshow(rgb0)

# RGB to gray
gray0 = rgb0 @ [0.2126, 0.7152, 0.0722]
plt.imshow(gray0, cmap = 'gray')

# Normalize
gray0 = gray0/255

# use Canny as edge detector
edges = feature.canny(gray0, low_threshold = 0, high_threshold = .85)
plt.imshow(edges, cmap = 'gray')

# Severly dilate
dilated = binary_dilation(edges, selem=morphology.diamond(50), out=None)
plt.imshow(dilated, cmap = 'gray')

# Apply mask to gray
gray1 = np.asarray(dilated)
gray1 = np.where(dilated, gray0, 0)
# gray1 = np.where(dilated[...,None], rgb0, 0)
plt.imshow(gray1, cmap = 'gray')




# Sharpening?

import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

blurred_f = gray1
     
filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
filter_blurred_f2 = ndimage.gaussian_filter(blurred_f, 8)
# filter_blurred_f2 = ndimage.gaussian_filter(blurred_f, 3, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

alpha = 30
sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
sharpened2 = blurred_f + alpha * (blurred_f - filter_blurred_f2)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(gray1, cmap = 'gray')
ax[0].set_title('Original Gray', fontsize=20)
ax[1].imshow(sharpened, cmap = 'gray')
ax[2].imshow(sharpened2, cmap = 'gray')

fig.tight_layout()
plt.show()

