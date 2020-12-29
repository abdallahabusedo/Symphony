import cv2
from scipy.ndimage import binary_closing
from skimage import io
import os
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_local, threshold_otsu
from skimage.measure import find_contours


def get_fname_images_tuple(directory):
    fnames = os.listdir(directory)
    to_return = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = (rgb2gray(io.imread(path)) * 255).astype(np.uint8)
        to_return.append((fn, gray_scale_image))

    return to_return


def Local_Thresholding(original_image):
    block_size = 21
    # calculate the local threshold value
    threshold_local_value = threshold_local(original_image, block_size, offset=10)
    # apply the local threshold value on the image
    binary = original_image > threshold_local_value
    return binary


def lineRemover(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    result = cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    return result


def objectDetection(LineRemovedArray):
    kernel = np.ones((1, 1))
    gray = cv2.cvtColor(LineRemovedArray, cv2.COLOR_BGR2GRAY)
    binary_image = Local_Thresholding(gray)
    binary_image_temp = binary_image.astype(np.uint8)
    binary_image_temp = np.invert(binary_image_temp * 255)
    imgDial = cv2.dilate(binary_image_temp, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # LineRemovedArray = cv2.cvtColor(LineRemovedArray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(LineRemovedArray, contours, -1, (0, 255, 0), 1)
    objectDRow = (LineRemovedArray)
    return objectDRow


def divide(img):
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
    binary = cv2.threshold(img, 215, 255, type=cv2.THRESH_BINARY)[1]
    dilation = cv2.dilate(binary, kernel, iterations=50)
    Img_h, Img_w = img.shape
    lines = find_contours(dilation, 0.8)

    results = []
    for c in lines:
        ll, ur = np.min(c, 0), np.max(c, 0)  # getting the two points
        wh = ur - ll  # getting the width and the height
        (x, y, w, h) = ll[0], ll[1], wh[1], wh[0]
        # getting the 4 contours that we have (4 groups of the numbers)
        results.append((x, y, w, h))

    line_positions = results
    for box in results:
        X, Y, width, height = box
        cv2.rectangle(dilation, (int(Y), int(X)), (int(Y + width), int(X + height)), (0, 255, 0), 10)

    contours = find_contours(dilation, 0.8)
    results = []
    for c in contours:
        ll, ur = np.min(c, 0), np.max(c, 0)  # getting the two points
        wh = ur - ll  # getting the width and the height
        (x, y, w, h) = ll[0], ll[1], wh[1], wh[0]
        # getting the 4 contours that we have (4 groups of the numbers)
        results.append((x, y, w, h))

    i = 1
    xup = 0
    l = len(results)
    ROWSImages = []
    for box in results:
        X, Y, width, height = box
        if i == l:
            xl = Img_h
        else:
            xl, yl, widthl, heightl = results[i]
        cv2.rectangle(dilation, (int(Y), int(X)), (int(Y + width), int(X + height)), (0, 255, 0), 1)
        Image = img[int(X - (X - xup) / 2):int(X + height + ((xl - X) / 2)), 0:int(Img_w), ]  # Y-50
        ROWSImages.append(Image)
        i = i + 1
        xup = X + height
    return binary


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = np.sum(pts,axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def Corners_Detection(binary_Image,Image):
    # detecting corners:
    kernel = np.ones((5, 5))
    binary_image_temp = binary_Image.astype(np.uint8)
    binary_image_temp = np.invert(binary_image_temp * 255)
    imgDial = cv2.dilate(binary_image_temp, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    imgContours = Image.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = Image.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 1)
    io.imshow(imgContours)
    io.show()
    # areas_of_contours = []
    # for s in contours:
    #     areas_of_contours.append(cv2.contourArea(s))
    #
    # averageArea = np.sum(areas_of_contours) / (len(areas_of_contours))
    #
    # areas = []
    # new_contours = []
    # for i in range(len(areas_of_contours)):
    #     if (areas_of_contours[i] >= averageArea):
    #         areas.append(areas_of_contours[i])
    #         new_contours.append(contours[i])
    #
    # minLL = np.min(new_contours[0], 0)
    # minLLX = 1000000
    # minLLY = 1000000
    # maxUR = np.max(new_contours[0], 0)
    # maxURX = 0
    # maxURY = 0
    #
    # for c in new_contours:
    #     [ll, ur] = np.min(c, 0), np.max(c, 0)  # getting the two points
    #     if ll[0][1] < minLLY:
    #         minLLY = ll[0][1]
    #     if ll[0][0] < minLLX:
    #         minLLX = ll[0][0]
    #     if ur[0][0] > maxURX:
    #         maxURX = ur[0][0]
    #     if ur[0][1] > maxURY:
    #         maxURY = ur[0][1]
    #     minLL[0][0] = minLLX
    #     minLL[0][1] = minLLY
    #     maxUR[0][0] = maxURX
    #     maxUR[0][1] = maxURY
    #     wh = maxUR - minLL  # getting the width and the height
    #     (x, y, w, h) = minLL[0][0], minLL[0][1], wh[0][0], wh[0][1]
    #     result = (x, y, w, h)
    # # When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
    # X, Y, width, height = result
    # cv2.rectangle(imgBigContour, (int(minLLX), int(minLLY)),
    #               (int(maxURX), int(maxURY)), (0, 255, 0), 2)
    #
    # for cont in new_contours:
    #     for i in range(len(cont)):
    #         if cont[i][0][1] == Y:
    #             x1 = cont[i][0][0]
    #         if cont[i][0][0] == X:
    #             y3 = cont[i][0][1]
    #         if cont[i][0][1] == Y + height:
    #             x4 = cont[i][0][0]
    #         if cont[i][0][0] == X + width:
    #             y2 = cont[i][0][1]
    #
    # # countour (rectangle)
    # xc1 = minLLX
    # yc1 = minLLY
    # xc2 = maxURX
    # yc2 = minLLY
    # xc3 = minLLX
    # yc3 = maxURY
    # xc4 = maxURX
    # yc4 = maxURY
    #
    # # of the image
    # y1 = yc1
    # x2 = xc2
    # x3 = xc3
    # y4 = yc4
    #
    # A = [[xc1, yc1, 1, 0, 0, 0, -xc1 * x1, -yc1 * x1],
    #      [0, 0, 0, xc1, yc1, 1, -xc1 * y1, -yc1 * y1],
    #      [xc2, yc2, 1, 0, 0, 0, -xc2 * x2, -yc2 * x2],
    #      [0, 0, 0, xc2, yc2, 1, -xc2 * y2, -yc2 * x2],
    #      [xc3, yc3, 1, 0, 0, 0, -xc3 * x3, -yc3 * x3],
    #      [0, 0, 0, xc3, yc3, 1, -xc3 * y3, -yc3 * y3],
    #      [xc4, yc4, 1, 0, 0, 0, -xc4 * x4, -yc4 * x4],
    #      [0, 0, 0, xc4, yc4, 1, -xc4 * y4, -yc4 * y4]]
    #
    # #b = [[x1], [y1], [x2], [y2], [x3], [y3], [x4], [y4]]
    # pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    # AT = np.transpose(A)
    # S = np.linalg.inv(np.dot(AT, A))
    # W = np.dot(AT, b)
    # HH = np.dot(S, W)
    #
    # h11 = HH[0][0]
    # h12 = HH[1][0]
    # h13 = HH[2][0]
    # h21 = HH[3][0]
    # h22 = HH[4][0]
    # h23 = HH[5][0]
    # h31 = HH[6][0]
    # h32 = HH[7][0]
    # h33 = 1
    #
    # dx = x1 - X
    # dy = y3 - Y
    #
    # s = np.sqrt(np.square(dx) + np.square(dy))
    # angle = np.arcsin(dy / s)
    # angle_degree = (angle / (22 / 7)) * 180
    #
    # matrix = np.array([
    #     [h11, h12, h13],
    #     [h21, h22, h23],
    #     [h31, h32, h33]])
    #
    # kernel2 = np.array([
    #     [0, 0, 0],
    #     [1, 1, 1],
    #     [0, 0, 0]], np.uint8)
    #
    # binary_image = binary_Image.astype(np.uint8)
    # binary_image_temp = np.invert(binary_image * 255)
    # erosin_image = cv2.erode(binary_image_temp, kernel2, iterations=15)
    # image_histogram = cv2.calcHist([erosin_image], [0], None, [256], [0, 256])
    #
    # tf_img = Image
    # if (image_histogram[255][0] < 30):
    #     tform = cv2.transform.ProjectiveTransform(matrix=matrix)
    #     tf_img = cv2.transform.warp(Image, tform)
    # return pts


image = io.imread('6.png',as_gray="true")
binary_image = Local_Thresholding(image)
Corners_Detection(binary_image ,image)
#im =four_point_transform(image,res)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#result = divide(gray)
#result3 = lineRemover(result[0])
# result = objectDetection(result)

