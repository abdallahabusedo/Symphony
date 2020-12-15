import cv2
from skimage.filters import threshold_otsu, threshold_local
from skimage import io
from skimage import data,transform,img_as_float
import math
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage, misc
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
from skimage.measure import find_contours
import numpy as np
from PIL import Image as im

def bwlabel(binary_image, param):
    pass


Image_path = "noteR.jpeg"
# to read the image in a grey scale mode
Image = cv2.imread(Image_path, 0)
image_width, image_height = Image.shape

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
# io.imshow(binary_image)
# io.show()



# detecting corners:

kernel = np.ones((5, 5))
binary_image_temp = binary_image.astype(np.uint8)
binary_image_temp  = np.invert(binary_image_temp*255)
imgDial = cv2.dilate(binary_image_temp, kernel, iterations=2) # APPLY DILATION
imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
#now we find contours
## FIND ALL COUNTOURS
imgContours = Image.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = Image.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 1)
# io.imshow(imgContours)
# io.show()

max = 0
for c in contours :
    [ll, ur] = np.min(c, 0), np.max(c, 0) #getting the two points
    wh = ur - ll  #getting the width and the height
    (x,y,w,h) = ll[0][0], ll[0][1], wh[0][0], wh[0][1]
    temp = w*h
    if(temp > max):
        max = temp
        result= (x,y,w,h)
        cont = c

#print(cont[pointnum][0][0 or 1])


#When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
X, Y, width, height = result
cv2.rectangle(imgBigContour, (int(X), int(Y)), (int(X+width), int(Y+height)), (0, 255, 0), 2)

for i in range(len(cont)) :
   if cont[i][0][1] == Y:
       x1 = cont[i][0][0]
   if cont[i][0][0] == X:
       y3 = cont[i][0][1]
   if cont[i][0][1] == Y+height:
       x4 = cont[i][0][0]
   if cont[i][0][0] == X+width:
       y2 = cont[i][0][1]
#countour (rectangle)
xc1= X
yc1 = Y
xc2 = X+width
yc2 = Y
xc3 = X
yc3 = Y+height
xc4 = X+width
yc4 = Y+height

#of the image
y1 = yc1
x2 = xc2
x3 = xc3
y4 = yc4



A = [[ xc1, yc1, 1, 0, 0, 0, -xc1*x1, -yc1*x1],
     [0, 0, 0, xc1, yc1, 1, -xc1*y1, -yc1*y1],
     [xc2 , yc2, 1, 0,0,0, -xc2*x2, -yc2*x2],
     [0,0,0, xc2, yc2, 1, -xc2*y2, -yc2*x2],
     [xc3, yc3, 1, 0,0,0, -xc3*x3, -yc3*x3],
     [0,0,0, xc3, yc3, 1, -xc3*y3, -yc3*y3],
     [xc4, yc4, 1,0,0,0,-xc4*x4, -yc4*x4],
     [0,0,0,xc4,yc4,1, -xc4*y4, -yc4*y4]]

b = [ [x1],
      [y1],
      [x2],
      [y2],
      [x3],
      [y3],
      [x4],
      [y4]]

AT = np.transpose(A)
S = np.linalg.inv(np.dot(AT,A))
W = np.dot(AT,b)
HH = np.dot(S,W)

# HH = np.dot(np.linalg.inv(A),b)

# HH2 = np.dot(np.linalg.inv(A),b)
# print(HH2)
h11 = HH[0][0]
h12 = HH[1][0]
h13 = HH[2][0]
h21 = HH[3][0]
h22 = HH[4][0]
h23 = HH[5][0]
h31 = HH[6][0]
h32 = HH[7][0]
h33 = 1



dx = x1-X
dy = y3-Y

s = np.sqrt(np.square(dx)+ np.square(dy))
angle = np.arcsin(dy/s)
angle_degree = (angle/(22/7))*180

rotated_array_image = scipy.ndimage.rotate(np.array(Image),angle=angle_degree)
#rotated_image = im.fromarray(rotated_array_image, 'RGB')
# imag = im.open(Image_path)
# imag.rotate(180-(angle_degree+100)).show()

# imgWarpColored = cv2.warpPerspective(Image, HH, (w, h))

# pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
# pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
# io.imshow(imgBigContour)
# io.show()
# print(image_width,image_height)
# new_image =np.ones((image_width,image_height))
# for x in range(image_width-1):
#     for y in range(image_height-1):
#         newx = (h11*x+h12*y+h13)/(h31*x+h32*y+1)
#         newy = (h21*x+h22*y+h23)/(h31*x+h32*y+1)
#         if newx>=image_width or newy>=image_height:
#             continue
#         new_image[int(newx),int(newy)] = Image[x,y]
#
#
# new_image = new_image.astype(np.uint8)
# # io.imshow(new_image)
# # io.show()

matrix = np.array([ [h11,h12,h13],
    [h21,h22,h23],
    [h31,h32,h33]
])

tform = transform.ProjectiveTransform(matrix=matrix)
tf_img = transform.warp(Image,tform)
io.imshow(tf_img)
io.show()
