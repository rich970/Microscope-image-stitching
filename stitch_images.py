#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:18:58 2020
Beta version test test
@author: richard
"""

import numpy as np
import Stoner
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import os
import re
from skimage.feature import match_template

# from PyQt5.QtWidgets import (QWidget, QSlider,
#                              QLabel, QApplication)
# from PyQt5.QtCore import Qt

# =============================================================================
# Functions:
# =============================================================================
# class ImageStitcher(QWidget):

#     def __init__(self):
#         super().__init__()
#         self.initUI()

#     def initUI(self):
#         blend_sld = QSlider(Qt.Horizontal, self)

def onselect(eclick, erelease):
    "eclick and erelease are matplotlib events at press and release."
    print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
    print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
    print('used button  : ', eclick.button)


def toggle_selector(event):
    print('Key pressed.')
    if event.key in 'enter' and toggle_selector.RS.active:
        print('RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print('RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

def sort_files(files):
    file_no = []
    sortedfiles = [None] * len(files)
    for f in files:
        file_no = re.findall('_(\d*)\.', f)
        ind = int(file_no[0])
        sortedfiles[ind] = f
    return sortedfiles
# =============================================================================
# User variables:
# =============================================================================
path = 'data/01092020 cfn5_10/'
fr = 250  # Width of frame to add in pixels
nc = 4  # Number of images stacked across (along columns)
nr = 3  # Number of images stacked vertically (along rows)
threshold = 85 # Threshold for the area calculation at the end
blending_level = 0.6 # Scaling factor for the blended regions between images


#def main():
# make a list of all our unprocessed image files
f = Stoner.DataFolder(path, pattern='*.jpg*')
files = sort_files(f.files) # sort ascending for looping
print(files)
plt.close('all')
fig1, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 6))
ax1.set_title('Primary image')
ax2.set_title('Image to join')
ax3.set_title('Sampled area')

# Primary picture is the first image which a subsequent image joins to
PrimPic = cv.imread(path+files[0])
PrimPic = cv.cvtColor(PrimPic, cv.COLOR_BGR2GRAY)
# Size of single images, vertically - rows (sr) and across - columns (sc)
sr, sc = np.shape(PrimPic)

# Main picture is the total stitched image.
# For now this is empty space for all images, plus a frame on both sides
MainPic = np.zeros([2*fr+(nr*sr), 2*fr+(nc*sc)])
MainPic[fr:sr+fr, fr:sc+fr] = PrimPic

# define the edge of the PrimPic
redge = fr+sr
cedge = fr+sc
a = 1
usr_var = str()
while a < len(files):
    plt.close(2)
    ax2.cla()
    ax3.cla()
    StitchingPic = cv.imread(path+files[a])
    StitchingPic = cv.cvtColor(StitchingPic, cv.COLOR_BGR2GRAY)

    # Level the images so mean pixel intensity is 128
    PrimPic = cv.add(PrimPic, 128 - np.mean(PrimPic))
    StitchingPic = cv.add(StitchingPic, 128 - np.mean(StitchingPic))

    ax1.imshow(PrimPic, cmap='gray')
    ax2.imshow(StitchingPic, cmap='gray')
    plt.show()
    toggle_selector.RS = RectangleSelector(
        ax2, onselect, drawtype='box', interactive=True)

    while toggle_selector.RS.active == True:
        i, j, k, l = toggle_selector.RS.extents
        Sample = StitchingPic[int(k):int(l), int(i):int(j)]
        ax3.imshow(Sample, cmap='gray')
        plt.connect('key_press_event', toggle_selector)
        plt.pause(0.4)

    cross_cor = match_template(PrimPic, Sample)
    ind = np.unravel_index(np.argmax(cross_cor), cross_cor.shape)
    # x, y coords of the centre of the matched region on the main pic
    x, y = ind[::-1]

    ax3.set_title('Matched sample in primary image')
    # highlight matched region
    hsample, wsample = Sample.shape

    rect = plt.Rectangle(
        (x, y), wsample, hsample, edgecolor='r', facecolor='none')
    ax3.imshow(PrimPic, cmap='gray')
    # Plot a rectangle the same size as selected area on the matched region of MainPic
    ax3.add_patch(rect)
    plt.pause(0.2)
    # Check to see if the user is happy with the mathced region
    usr_var = input('Is the image matched ok? [y,n]\n')
    if usr_var == str('n'):
        continue
    if usr_var == str('y'):
        a += 1
    else:
        print('Type either \'y\' or \'n\':  assuming \'n\'')
        continue
    # Calculate the required offest for the stitched image
    delta_r = y-int(k)  # (sr-y) + int(k)
    delta_c = x-int(i)  # int(i) - x

    rmin = (redge-sr)+delta_r
    rmax = (redge)+delta_r
    cmin = (cedge-sc)+delta_c
    cmax = (cedge)+delta_c

    # Generate a picture same size as main pic to overlay but with
    # the stitching pic with the required offset.
    OverlayPic = np.zeros([2*fr+(nr*sr), 2*fr+(nc*sc)])
    OverlayPic[rmin:rmax, cmin:cmax] = StitchingPic
    fig4, ax4 = plt.subplots(figsize=(6, 8))
    ax4.set_title('Stitched image')
    # MainPic[rmin:rmax, cmin:cmax] += StitchingPic
    # Images are in correct relative positions. Find overlapping pixels.
    Mask = cv.bitwise_and(MainPic, OverlayPic)
    ind = np.where(Mask > 0)
    # Half the value of the overlapping pixels ready for the images to be added
    MainPic[ind] = MainPic[ind]*blending_level
    OverlayPic[ind] = OverlayPic[ind]*blending_level

    # Add the two images to create
    MainPic = cv.addWeighted(MainPic, 1.0, OverlayPic, 1.0, 0)
    ax4.imshow(MainPic, cmap='gray')

    PrimPic = StitchingPic  # Previous Stitching pic becomes next Primary Pic
    # Reset the image edge locations ready for the next image
    redge = rmax
    cedge = cmax

    plt.pause(2)
    plt.figure(1)
    plt.waitforbuttonpress()

_, ThresholdPic = cv.threshold(MainPic, threshold, 255, cv.THRESH_BINARY)
plt.imshow(ThresholdPic, cmap='gray')
area = np.sqrt(np.sum(ThresholdPic)/255)  # Area in pixels**2
print('Area = {0:.0f} pixels^2'.format(area))


#if __name__ == '__main__':
 #   main()