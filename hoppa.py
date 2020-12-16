#!/usr/bin/env python
# coding: utf-8

# In[116]:


from commonfunctions import *
import numpy as np
import cv2
import math
import skimage
import scipy.ndimage
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk

# Show the figures / plots inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[173]:


img = cv2.imread(r'G:\Capture.jpg')
kernel =  np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
],np.uint8)
kernel2 =  np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
],np.uint8)
_, bina = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations =2)

finimg= (img) - (dilation)
hoppa= 255-finimg
dil2 = cv2.dilate(hoppa,kernel,iterations =1)
ero2 = cv2.erode(dil2,kernel,iterations = 1)

show_images([img,bina,hoppa,ero2])


# In[180]:


####get staff lines by horizontal projections
img = cv2.imread(r'G:\Capture.jpg')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#GaussianFilter= cv2.GaussianBlur(gray, (5,5), 0)
_, binarizedImage = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binarizedImage[binarizedImage == 0] = 1
binarizedImage[binarizedImage == 255] = 0
horizontal_projection = np.sum(binarizedImage, axis=1);
height, width = binarizedImage.shape
blankImage = np.zeros((height, width, 3), np.uint8)
for row in range(height):
    cv2.line(blankImage, (0,row), (int(horizontal_projection[row]*width/height),row), (255,255,255), 1)
show_images([img,binarizedImage,blankImage])


# In[ ]:





# In[ ]:





# In[ ]:




