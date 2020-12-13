import cv2
from skimage.filters import gaussian, threshold_otsu
import skimage.io as io
path = "lool.png"
img = cv2.imread(path, 0)  # to read the image in a grey scale mode

img_with_gaussian_filtered = gaussian(img,sigma=8);
cv2.cv2.imwrite('filtered.png', img_with_gaussian_filtered*255)

# handle poop lightnening: compare each pixel in the original image by the corresponding pixel in the filtered image
# in order to apply thresholding

thresh = threshold_otsu(img)

# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         original_pixel = img.item(i, j)
#         filtered_pixel = img_with_gaussian_filtered.item(i, j)
#         print(original_pixel - filtered_pixel*255)
#         if original_pixel - filtered_pixel > thresh:
#             img[i,j] = 255;
#         else:
#             img[i,j] = 0;

# thresh = threshold_otsu(img)
# print(thresh)
# binary = img < thresh
# cv2.cv2.imwrite('thresholded.png', binary*255)

retval, img_Er = cv2.threshold(img,thresh,255,type=cv2.THRESH_BINARY)
io.imshow(img_Er)
io.show()
