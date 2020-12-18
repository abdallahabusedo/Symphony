import cv2
from matplotlib.pyplot import bar
from skimage.filters import threshold_local
from skimage import io
from skimage import transform
import scipy
from scipy import ndimage
import numpy as np
from skimage.measure import find_contours
import scipy.ndimage
import matplotlib.pyplot as plt
from math import ceil
from skimage.color import rgb2gray
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin


# get staff lines by horizontal projections
def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------------
# this function divides the music sheet to sub images each containing a row of the notes
# -----------------------------------------------------------------------------------------------------------------------------------------


def divide(output_path, img):
    kernel = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ], np.uint8)
    kernel2 = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], np.uint8)
    # ---------------------------dilates the image to get the staff lines only-------------------------------------------------------------
    retval, img_binary = cv2.threshold(img, 215, 255, type=cv2.THRESH_BINARY)
    dilation = cv2.dilate(img_binary, kernel, iterations=50)
    Img_h, Img_w = img.shape
# ---------------------------finds the contoours that surround the lines ------------------------------------------------------------
    rect = cv2.getStructuringElement(
        cv2.MORPH_RECT, (60, 50))  # the structure element
    img_closing = binary_closing(dilation, rect)
    contours = find_contours(dilation, 0.8)
    results = []

    for c in contours:
        ll, ur = np.min(c, 0), np.max(c, 0)  # getting the two points
        wh = ur - ll  # getting the width and the height
        (x, y, w, h) = ll[0], ll[1], wh[1], wh[0]
        # getting the 4 contours that we have (4 groups of the numbers)
        results.append((x, y, w, h))
    # When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
    line_positions = results
    for box in results:
        X, Y, width, height = box
        cv2.rectangle(dilation, (int(Y), int(X)),
                      (int(Y+width), int(X+height)), (0, 255, 0), 10)
# ---------------------------finds the main big contours to cut the image-------------------------------------------------------------
    img_closing = binary_closing(dilation, rect)
    contours = find_contours(dilation, 0.8)
    results = []
    for c in contours:
        ll, ur = np.min(c, 0), np.max(c, 0)  # getting the two points
        wh = ur - ll  # getting the width and the height
        (x, y, w, h) = ll[0], ll[1], wh[1], wh[0]
        # getting the 4 contours that we have (4 groups of the numbers)
        results.append((x, y, w, h))
    # When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
    i = 1
    xup = 0
    l = len(results)
    for box in results:
        X, Y, width, height = box
        if i == l:
            xl = Img_h
        else:
            xl, yl, widthl, heightl = results[i]
        cv2.rectangle(dilation, (int(Y), int(X)),
                      (int(Y+width), int(X+height)), (0, 255, 0), 1)
        Image = img[int(X-(X-xup)/2):int(X+height+((xl-X)/2)),
                    0:int(Img_w), ]  # Y-50
        cv2.imwrite((output_path+str(i)+".bmp"), Image)
        i = i+1
        xup = X+height
    return l, line_positions

# ------------------------------------------------------------------------------------------------------------
    # removing staff lines
    # for more clear img ,uncomment binarization


def remove_lines(out_path, in_path, img_count):
    kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8)
    kernel2 = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], np.uint8)
    for i in range(1, img_count):
        img = cv2.imread((in_path+str(i)+".bmp"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        erosion = cv2.erode(gray, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel2, iterations=1)
        cv2.imwrite((out_path+str(i)+".bmp"), dilation)


Image_path = "note2.png"
# to read the image in a grey scale mode
Image = cv2.imread(Image_path, 0)
image_width, image_height = Image.shape

# handle poop lightning: by applying local thresholding on the image
# in order to apply thresholding
# calculating the block size
# ------------------------to do make it general to all images------------------------#
block_size = 21
# calculate the local threshold value
threshold_local_value = threshold_local(Image, block_size, offset=10)
# apply the local threshold value on the image
binary_image = Image > threshold_local_value


def Corners_Detection(binary_Image):
    # detecting corners:
    kernel = np.ones((5, 5))
    binary_image_temp = binary_Image.astype(np.uint8)
    binary_image_temp = np.invert(binary_image_temp * 255)
    imgDial = cv2.dilate(binary_image_temp, kernel,
                         iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    return imgThreshold


imgThreshold = Corners_Detection(binary_image)

# now we find contours
# FIND ALL COUNTOURS
imgContours = Image.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = Image.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(
    imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 1)

areas_of_contours = []
for s in contours:
    areas_of_contours.append(cv2.contourArea(s))

averageArea = np.sum(areas_of_contours) / (len(areas_of_contours))

areas = []
new_contours = []
for i in range(len(areas_of_contours)):
    if(areas_of_contours[i] >= averageArea):
        areas.append(areas_of_contours[i])
        new_contours.append(contours[i])

minLL = np.min(new_contours[0], 0)
minLLX = 1000000
minLLY = 1000000
maxUR = np.max(new_contours[0], 0)
maxURX = 0
maxURY = 0

for c in new_contours:
    [ll, ur] = np.min(c, 0), np.max(c, 0)  # getting the two points
    if ll[0][1] < minLLY:
        minLLY = ll[0][1]
    if ll[0][0] < minLLX:
        minLLX = ll[0][0]
    if ur[0][0] > maxURX:
        maxURX = ur[0][0]
    if ur[0][1] > maxURY:
        maxURY = ur[0][1]
    minLL[0][0] = minLLX
    minLL[0][1] = minLLY
    maxUR[0][0] = maxURX
    maxUR[0][1] = maxURY
    wh = maxUR - minLL  # getting the width and the height
    (x, y, w, h) = minLL[0][0], minLL[0][1], wh[0][0], wh[0][1]
    result = (x, y, w, h)


# When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
X, Y, width, height = result
cv2.rectangle(imgBigContour, (int(minLLX), int(minLLY)),
              (int(maxURX), int(maxURY)), (0, 255, 0), 2)

for cont in new_contours:
    for i in range(len(cont)):
        if cont[i][0][1] == Y:
            x1 = cont[i][0][0]
        if cont[i][0][0] == X:
            y3 = cont[i][0][1]
        if cont[i][0][1] == Y+height:
            x4 = cont[i][0][0]
        if cont[i][0][0] == X+width:
            y2 = cont[i][0][1]

#countour (rectangle)
xc1 = minLLX
yc1 = minLLY
xc2 = maxURX
yc2 = minLLY
xc3 = minLLX
yc3 = maxURY
xc4 = maxURX
yc4 = maxURY

# of the image
y1 = yc1
x2 = xc2
x3 = xc3
y4 = yc4

A = [[xc1, yc1, 1, 0, 0, 0, -xc1*x1, -yc1*x1],
     [0, 0, 0, xc1, yc1, 1, -xc1*y1, -yc1*y1],
     [xc2, yc2, 1, 0, 0, 0, -xc2*x2, -yc2*x2],
     [0, 0, 0, xc2, yc2, 1, -xc2*y2, -yc2*x2],
     [xc3, yc3, 1, 0, 0, 0, -xc3*x3, -yc3*x3],
     [0, 0, 0, xc3, yc3, 1, -xc3*y3, -yc3*y3],
     [xc4, yc4, 1, 0, 0, 0, -xc4*x4, -yc4*x4],
     [0, 0, 0, xc4, yc4, 1, -xc4*y4, -yc4*y4]]

b = [[x1], [y1], [x2], [y2], [x3], [y3], [x4], [y4]]

AT = np.transpose(A)
S = np.linalg.inv(np.dot(AT, A))
W = np.dot(AT, b)
HH = np.dot(S, W)

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

s = np.sqrt(np.square(dx) + np.square(dy))
angle = np.arcsin(dy/s)
angle_degree = (angle/(22/7))*180


matrix = np.array([
    [h11, h12, h13],
    [h21, h22, h23],
    [h31, h32, h33]])

kernel2 = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]], np.uint8)

binary_image = binary_image.astype(np.uint8)
binary_image_temp = np.invert(binary_image*255)
erosin_image = cv2.erode(binary_image_temp, kernel2, iterations=15)
image_histogram = cv2.calcHist([erosin_image], [0], None, [256], [0, 256])


tf_img = Image
if(image_histogram[255][0] < 30):
    tform = transform.ProjectiveTransform(matrix=matrix)
    tf_img = transform.warp(Image, tform)

division_out_path = "cut"
lines_removed_out_path = "lines_removed"

# divide the image to small images containing each row
img_count, line_positions = divide(division_out_path, tf_img)
remove_lines(lines_removed_out_path, division_out_path, img_count)

io.imshow(img_count)
io.show()
