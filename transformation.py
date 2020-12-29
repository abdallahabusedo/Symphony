import cv2
import numpy as np
import skimage
from skimage import io
from skimage.transform import hough_line
from thresholding import *
import imutils



def getAngel(img):
    img = 255 - img
    tested_angle = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(img, theta=tested_angle)
    ind = np.unravel_index(np.argmax(h, axis=None), h.shape)
    degree = np.rad2deg(theta[ind[1]])
    return degree


image = io.imread('2.png', as_gray="true")
bi = Local_Thresholding(image)
angle = getAngel(bi)
print(angle)
rot = imutils.rotate(image, angle=angle)
io.imshow(rot)
io.show()
