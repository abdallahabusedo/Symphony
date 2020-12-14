import cv2
from skimage.filters import gaussian ,threshold_otsu, threshold_local
from skimage import data, io
import numpy as np
path = "note2.png"
img = cv2.imread(path, 0)  # to read the image in a grey scale mode
img_with_gaussian_filtered = gaussian(img,sigma=8);
#cv2.cv2.imwrite('filtered.png', img_with_gaussian_filtered*255)

# handle poop lightning: compare each pixel in the original image by the corresponding pixel in the filtered image
# in order to apply thresholding
# calculating the block size
block_size = 41
# calculate the local threshold value
threshold_local_value  = threshold_local(img ,block_size, offset=10)
# apply the local threshold value on the image
binary_image = img > threshold_local_value

# show the images
io.imshow(binary_image)
io.show()

