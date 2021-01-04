#the main for one image
from networkx.drawing.tests.test_pylab import plt
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
from skimage.morphology import dilation, closing, erosion, remove_small_objects, disk

from transformation import *
from thresholding import *
from ReadWrite import *
from preprocessing import *
import time
##############################
start = time.time()
Original_image = io.imread(r'E:\3rd year\image proc\project\Symphony\inputdata\test 2\02.PNG')


##to modify the light in the image
b = -30. # brightness
c = 190.  # contrast
Original_image = cv2.addWeighted(np.uint8(Original_image), 1. + c/127., np.uint8(Original_image), 0, b-c)

################rotating
Rotate_image = our_rotate(np.asarray(Original_image))
#################Thresholding
thresholed_rotated = thresholding(Rotate_image)
_,_, Rows_images = divide(np.uint8(thresholed_rotated))

# cv.imshow("cut",Rows_images[0])
# cv.waitKey(0)

Image.fromarray(Rows_images[2]).save("out.png")

bwArray = np.array(thresholed_rotated)
picWidth = len(bwArray[0])

horzPicCount = horizontalProjection(bwArray)

lineArray = getLines(horzPicCount, picWidth)

lineThickness, newLineArray = findBarLineWidth(lineArray)

lineArray = newLineArray

print(lineArray)
spaceSize, spaceBetweenBars = findSpacesSize(lineArray, lineThickness)

removed_line_pic = removeMe(thresholed_rotated, lineArray, lineThickness)


# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start}")


##############prespect
# kernel = np.array([[0, 1, 0],
#                    [1, 1, 1],
#                    [0, 1, 0]], np.uint8)
#
# kernel2 = np.array([[1, 1, 1],
#                    [1, 1, 1],
#                    [1, 1, 1]], np.uint8)
# thresh_dilated = 255-(cv2.dilate(255-np.uint8(thresholed_rotated), kernel, iterations=5))
# # thresh_erode = 255-(cv2.erode(255-np.uint8(thresh_dilated), kernel2, iterations=8))
#
# output = np.ones_like(Original_image)
#
# imgContours = Original_image.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
#
# contours = find_contours(thresh_dilated, 0.8)
# results = []
# for c in contours :
#     ll, ur = np.min(c, 0), np.max(c, 0) #getting the two points
#     wh = ur - ll  #getting the width and the height
#     (x,y,w,h) = ll[0], ll[1], wh[1], wh[0]
#     if w >= 300:
#       results.append((x,y,w,h)) #getting the 4 contours that we have (4 groups of the numbers)
#
# #When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
# for box in results:
#     X, Y, width, height = box
#     cv2.rectangle(np.uint8(thresh_dilated), (int(Y), int(X)), (int(Y+width), int(X+height)), (0, 0, 0), 5)
#
# lines = getlines(255-np.uint8(Rotate_image))
# wantedLines = []
# print(lines)
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     if abs(y1-y2) <= 50 and x1!=0:
#         wantedLines.append(line)
#         #cv2.line(255-Rotate_image, (x1, y1), (x2, y2), (0, 0, 0), 3)


# Image.fromarray(255-thresh_erode).save("out.png")
#############################
# result = divide(rgb2gray(rot_thresholded))
# io.imshow(result[0].astype('uint8'))
# io.show()
# image = io.imread('26.jpg',as_gray="true")
# binary_image = Local_Thresholding(image)
# res = getFourCorners(binary_image)
# #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #result = divide(gray)
# #result3 = lineRemover(result[0])
# #result = objectDetection(result)
# io.imshow(rot_thresholded.astype('uint8'))
# io.show()