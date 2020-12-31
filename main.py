
#the main for one image

import cv2
from skimage import io
import os
import numpy as np
from skimage.color import rgb2gray
from skimage.measure import find_contours
from transformation import *

Original_image = io.imread('26.jpg')

#rotating the image (if it needs rotation)
Rotate_image = our_rotate(Original_image)
# io.imshow(Rotate_image)
# io.show()











# image = io.imread('26.jpg',as_gray="true")
# binary_image = Local_Thresholding(image)
# res = getFourCorners(binary_image)
# #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #result = divide(gray)
# #result3 = lineRemover(result[0])
# #result = objectDetection(result)
# io.imshow(res)
# io.show()