import cv2
from skimage.filters import threshold_otsu, threshold_local
from skimage import io
import numpy as np
def bwlabel(binary_image, param):
    pass

Image_path = "noteR.jpeg"
# to read the image in a grey scale mode
Image = cv2.imread(Image_path, 0)

# handle poop lightning: by applying local thresholding on the image
# in order to apply thresholding
# calculating the block size
# ------------------------to do make it general to all images------------------------#
block_size = 21
# calculate the local threshold value
threshold_local_value  = threshold_local(Image ,block_size, offset=10)
# apply the local threshold value on the image
binary_image = Image > threshold_local_value

# show the images
io.imshow(binary_image)
io.show()


