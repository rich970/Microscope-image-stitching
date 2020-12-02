#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 19:39:34 2020

@author: richard

"""

import sys
import cv2 as cv
from PyQt5 import QtWidgets, QtCore # import PyQt5 before matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from skimage.feature import match_template
import numpy as np


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig1 = Figure(figsize=(width, height),
                      dpi=dpi,
                      facecolor='gray')
        self.ax = fig1.subplots(nrows=2, ncols=3, squeeze=True)
        fig1.subplots_adjust(hspace=0.05, wspace=0.01,
                             left=0.02, right=0.98,
                             bottom=0.02, top=0.98)
        self.clean_figs()

        super().__init__(fig1)

    def clean_figs(self):
        self.ax[0,0].set_title('Primary image')
        self.ax[0,0].set_aspect(0.75)
        self.ax[0,1].set_title('Image to join')
        self.ax[0,1].set_aspect(0.75)
        self.ax[0,2].set_title('Matched area')
        self.ax[0,2].set_aspect(0.75)
        self.ax[1,0].set_title('Sampled Area')
        self.ax[1,1].set_title('Stitched image')
        self.ax[1,2].set_title('Thresholded image')

        for axes in self.ax.flatten():
            axes.tick_params(length=0)
            axes.set_xticklabels([])
            axes.set_yticklabels([])


class ImageStitcher(QtWidgets.QMainWindow, QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        
        self.canvas = MplCanvas(self, width=5, height=5, dpi=100)
        
        QBtn = QtWidgets.QDialogButtonBox.Open
        self.button_file = QtWidgets.QDialogButtonBox(QBtn)
        self.button_file.clicked.connect(self.getfile)

        QBtn = QtWidgets.QDialogButtonBox.Save
        self.button_save= QtWidgets.QDialogButtonBox(QBtn)
        self.button_save.clicked.connect(self.savefile)
        
        # Just some button connected to `update_image` method
        self.button_img = QtWidgets.QPushButton('Update images')
        self.button_img.clicked.connect(self.update_images)
        
         # Disable rectangle selector
        self.button_RS = QtWidgets.QPushButton('Rectangle selector')
        self.button_RS.setCheckable(True) 
        self.button_RS.clicked.connect(self.toggle_RS)
        
        # Button for calling the match template algorithm
        self.button_match = QtWidgets.QPushButton('Match!') 
        self.button_match.clicked.connect(self.match_and_plot)
        
        # Button for calling the stitching  algorithm
        self.button_stitch = QtWidgets.QPushButton('Stitch!') 
        self.button_stitch.clicked.connect(self.stitch_and_plot)
        
        # Button for moving onto next image
        self.button_next = QtWidgets.QPushButton('Next!') 
        self.button_next.clicked.connect(self.next_images)
        
        self.line_nr = QtWidgets.QLineEdit()
        self.line_nc = QtWidgets.QLineEdit()
        layout_rowscols = QtWidgets.QHBoxLayout()
        layout_rowscols.addWidget(self.line_nr)
        layout_rowscols.addWidget(self.line_nc)
        self.label_n = QtWidgets.QLabel()
        self.label_n.setText('Image grid [Rows x Cols]: ')
        # Input dialogue for choosing image magnification
        self.combo_mag = QtWidgets.QComboBox()
        # Calibration factors for different magnifications [px/cm]
        self.mag_values = {'x4': 8800,
                            'x10' : 1,
                            'x20' : 1,
                            'x50' : 1}
        self.combo_mag.addItems(self.mag_values.keys())
        self.combo_mag.currentIndexChanged.connect(self.change_mag)
        self.line_mag = QtWidgets.QLineEdit()
        self.line_mag.setText('{0} [px/cm]'.format(self.mag_values['x4']))
        layout_mag = QtWidgets.QHBoxLayout()
        layout_mag.addWidget(self.combo_mag)
        layout_mag.addWidget(self.line_mag)
        self.label_mag = QtWidgets.QLabel()
        self.label_mag.setText('Magnification: ')
        # Slider for blending level between stitched images
        self.slider_blending = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_blending.setMinimum(0)
        self.slider_blending.setMaximum(100)
        self.slider_blending.setValue(60)
        self.label_blending = QtWidgets.QLabel()
        self.label_blending.setText('Blending level [%]: ')
        
        # Set the threshold pixel value for the area calculation
        self.slider_threshold = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_threshold.setMinimum(0)
        self.slider_threshold.setMaximum(100)
        self.slider_threshold.setValue(17)
        self.label_threshold = QtWidgets.QLabel()
        self.label_threshold.setText('Threshold level [%]: ')

        # Button for moving onto next image
        self.button_calcarea = QtWidgets.QPushButton('Calculate area') 
        self.button_calcarea.clicked.connect(self.calc_area)
        self.line_area = QtWidgets.QLineEdit()
        
        # set the layout
        layout = QtWidgets.QGridLayout()
        layout_sub = QtWidgets.QGridLayout()

    #layout_sub.setSpacing(0)
        layout.setRowStretch(0,5)
        layout.setRowStretch(1,1)
        layout_sub.setColumnStretch(0,1)
        layout_sub.setColumnStretch(1,1)
        layout_sub.setColumnStretch(2,1)
        layout_sub.setColumnStretch(3,1)

        layout_sub.addWidget(self.button_file, 0, 2, 1, 1)
        layout_sub.addWidget(self.button_img, 0, 3, 1, 1)
        layout_sub.addWidget(self.button_RS, 1, 0, 1, 1)
        layout_sub.addWidget(self.button_match, 1, 1, 1, 1)
        layout_sub.addWidget(self.button_stitch, 1, 2, 1, 1)
        layout_sub.addWidget(self.button_next, 1, 3, 1, 1)
        layout_sub.addWidget(self.label_mag, 2, 0, 1, 1)
        layout_sub.addLayout(layout_mag, 2, 1, 1, 1)
        layout_sub.addWidget(self.label_blending, 2, 2, 1, 1)
        layout_sub.addWidget(self.slider_blending, 2, 3, 1, 1)
        layout_sub.addWidget(self.label_n, 3, 0, 1, 1)
        layout_sub.addLayout(layout_rowscols, 3, 1, 1, 1)
        self.line_nr.setText("2.5")
        self.line_nc.setText("2.5")
        layout_sub.addWidget(self.label_threshold, 3, 2, 1, 1)
        layout_sub.addWidget(self.slider_threshold, 3, 3, 1, 1)
        self.slider_threshold.setValue(17)
        layout_sub.addWidget(self.button_calcarea, 4, 1, 1, 1)
        layout_sub.addWidget(self.line_area, 4, 2, 1, 1)
        self.line_area.setText("Area = {0} cm^2".format(0))
        layout_sub.addWidget(self.button_save, 4, 3, 1, 1)

        layout.addWidget(self.canvas)
        layout.addLayout(layout_sub, 1, 0, 1, 1)
        # Create a placeholder widget to hold our button and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.show()

    def change_mag(self):
        ind = self.combo_mag.currentText()
        mag_value = self.mag_values[ind]
        self.line_mag.setText('{0} [px/cm]'.format(mag_value))

    def update_images(self):
        nc = float(self.line_nc.text()) 
        nr = float(self.line_nr.text()) 

        file_iter = iter(self.fileName)

        self.PrimPic = cv.imread(next(file_iter))
        self.PrimPic = cv.cvtColor(
            self.PrimPic, cv.COLOR_BGR2GRAY)
        self.StitchingPic = cv.imread(next(file_iter))
        self.StitchingPic = cv.cvtColor(
            self.StitchingPic, cv.COLOR_BGR2GRAY)

        sr, sc = np.shape(self.PrimPic)
        self.sr = sr
        self.sc = sc
        self.nr = nr
        self.nc = nc
        self.file_iter = file_iter
        
        # define the edge of the PrimPic
        self.redge = int(sr*1.2)
        self.cedge = int(sc*1.2)
        
        rmin = (self.redge-self.sr)
        rmax = (self.redge)
        cmin = (self.cedge-self.sc)
        cmax = (self.cedge)
        
        # Main picture is the total stitched image.
        # For now this is empty space for all images, plus a frame
        self.MainPic = np.zeros([int(nr*sr), int(nc*sc)])
        self.MainPic[rmin:rmax, cmin:cmax] = self.PrimPic

        self.canvas.ax[0,0].imshow(self.PrimPic, cmap='gray')
        self.canvas.ax[0,1].imshow(self.StitchingPic, cmap='gray')
        self.canvas.draw()

    def next_images(self):
        try:
            self.StitchingPic = cv.imread(next(self.file_iter))
            self.StitchingPic = cv.cvtColor(
                self.StitchingPic, cv.COLOR_BGR2GRAY)
            
            for axes in self.canvas.ax.flatten():
                axes.cla()
            self.canvas.clean_figs()
            
            self.canvas.ax[0,0].imshow(self.PrimPic, cmap='gray')
            self.canvas.ax[0,1].imshow(self.StitchingPic, cmap='gray')
            self.canvas.draw()
        except StopIteration:
            print('All files have been stitched')
    
    def getfile(self):
         dlg = QtWidgets.QFileDialog()
         dlg.setFileMode(QtWidgets.QFileDialog.AnyFile)
         self.fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(
             self,"QFileDialog.getOpenFileNames()", 
             "","All Files (*)")
         if self.fileName:
             print(self.fileName)

    def savefile(self):
         dlg = QtWidgets.QFileDialog()
         dlg.setFileMode(QtWidgets.QFileDialog.AnyFile)
         self.saveName, _ = QtWidgets.QFileDialog.getSaveFileName(
             self,"QFileDialog.getSaveFileName()", 
             "","All Files (*)")
         if self.saveName:
             plt.imsave(self.saveName,
                        self.MainPic, format='png', cmap='gray')
             print('Saved file: {0}'.format(self.saveName))


    def onselect(self, eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        print(self.RS.extents)
        i, j, k, l = self.RS.extents
        self.Sample = self.StitchingPic[int(k):int(l), int(i):int(j)]
        self.canvas.ax[1,0].imshow(self.Sample, cmap='gray')

    def toggle_RS(self):
        if self.button_RS.isChecked():
            print('RectangleSelector activated.')
            self.RS = RectangleSelector(
                self.canvas.ax[0,1],
                self.onselect,
                drawtype='box',
                interactive=True)
        else:
            print('RectangleSelector deactivated.')
            self.RS.set_active(False)

    def match_and_plot(self):
        self.button_RS.setChecked(False) 
        self.toggle_RS()
        self.x, self.y = self.match_sample()
        hsample, wsample = self.Sample.shape
        rect = plt.Rectangle(
            (self.x, self.y), 
            wsample, hsample, 
            edgecolor='r', facecolor='none')
        self.canvas.ax[0,2].imshow(self.PrimPic, cmap='gray')
        # Plot a rectangle the same size as selected area on the matched region of MainPic
        self.canvas.ax[0,2].add_patch(rect)
        self.canvas.draw()
        
    def match_sample(self):
        
        cross_cor = match_template(self.PrimPic, self.Sample)
        ind = np.unravel_index(np.argmax(cross_cor), cross_cor.shape)
        # x, y coords of the centre of the matched region on the main pic
        print(ind[::-1])
        return ind[::-1]

    def stitch_and_plot(self):
       # fr = self.fr  # Width of frame to add in pixels
        sr = self.sr
        sc = self.sc
        nr = self.nr
        nc = self.nc
        
        val = int(self.slider_blending.value()) 
        blending_level = val/100 # Scaling factor for the blended regions between images

        # coordinates for edge of the PrimPic
        redge = self.redge
        cedge = self.cedge
        
        i, j, k, l = self.RS.extents
        delta_r = self.y-int(k)  # (sr-y) + int(k)
        delta_c = self.x-int(i)  # int(i) - x

        rmin = (redge-sr)+delta_r
        rmax = (redge)+delta_r
        cmin = (cedge-sc)+delta_c
        cmax = (cedge)+delta_c
        
        
        # Generate a picture same size as main pic to overlay but with
        # the stitching pic with the required offset.
        OverlayPic = np.zeros(np.shape(self.MainPic))
        OverlayPic[rmin:rmax, cmin:cmax] = self.StitchingPic
    
        
        # MainPic[rmin:rmax, cmin:cmax] += StitchingPic
        # Images are in correct relative positions. Find overlapping pixels.
        Mask = cv.bitwise_and(self.MainPic, OverlayPic)
        ind = np.where(Mask > 0)
        # Half the value of the overlapping pixels ready for the images to be added
        self.MainPic[ind] = self.MainPic[ind]*blending_level
        OverlayPic[ind] = OverlayPic[ind]*blending_level
    
        # Add the two images to create
        self.MainPic = cv.addWeighted(self.MainPic, 1.0, OverlayPic, 1.0, 0)
        self.canvas.ax[1,1].cla()
        self.canvas.clean_figs()
        self.canvas.ax[1,1].imshow(self.MainPic, cmap='gray')
        self.canvas.draw()
        # Previous Stitching pic becomes next Primary Pic
        self.PrimPic = self.StitchingPic  
        # Reset the image edge locations ready for the next image
        self.redge = rmax
        self.cedge = cmax

    def calc_area(self):
        val = int(self.slider_threshold.value()) 
        threshold = 255*(val/100) # Map this to a pixel value between 0-255
        _, ThresholdPic = cv.threshold(
            self.MainPic, threshold, 
            255, cv.THRESH_BINARY)

        self.canvas.ax[1,2].cla()
        self.canvas.clean_figs()
        self.canvas.ax[1,2].imshow(ThresholdPic, cmap='gray')
        self.canvas.draw()

        area = np.sqrt(np.sum(ThresholdPic)/255)  # Area in pixels**2
        print('Area = {0:.0f} pixels^2'.format(area))
        foo = self.line_mag.text()
        mag_cal = int(foo.split()[0])
        area = area/mag_cal
        self.line_area.setText("Area = {0:.3} cm^2".format(area))
        print('Area = {0:.3f} cm^2'.format(area))

app = QtWidgets.QApplication(sys.argv) 
w = ImageStitcher()
app.exec_()