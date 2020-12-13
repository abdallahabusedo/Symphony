import cv2
from skimage.filters import gaussian

path = "lool.png"
img = cv2.imread(path, 0)  # to read the image in a grey scale mode
img_with_gaussian_filtered = gaussian(img,sigma=8);
cv2.cv2.imwrite('filtered.png', img_with_gaussian_filtered*255)

# handle poop lightnening: compare each pixel in the original image by the corresponding pixel in the filtered image
# in order to apply thresholding

