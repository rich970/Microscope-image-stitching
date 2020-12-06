# Microscope-image-stitching
Semi-automatic stitching of microscope images obtained by raster scanning across the surface. The primary intention of this was for mapping thin-film samples and calculating the surface area. The script has now been superseeded by the GUI version.
# How it works:
- Users import a set of image files. The images should be ordered such that each subsequent image in the list has an overlapping region with the previous image. First image should be the top-left hand corner of the sample. 
- The first two images are displayed and the user selects a feature which occurs in both images. This is achieved with the Rectangle Selector widget of matplotlib.
- The selected feature is picked out in the primary image using a cross-correlation algorithm (match-template implemented in Skimage)
- If the user is happy, the two images are stitched together and the next two images are loaded.
- This repeat until all imported images have been stitched. 
- The final image is thresholded and the area is calculated.

# Requirments:
Sys, PyQt5, Numpy, cv2, maplotlib, Skimage.

# Feedback:
Feedback is very welcome! Either contact me directly or submit an issue. 

