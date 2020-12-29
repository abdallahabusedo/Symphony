import cv2
from skimage import io
import os
import numpy as np
from skimage.color import rgb2gray
from skimage.measure import find_contours

def get_fname_images_tuple(directory):
    fnames = os.listdir(directory)
    to_return = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = (rgb2gray(io.imread(path)) * 255).astype(np.uint8)
        to_return.append((fn, gray_scale_image))

    return to_return





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
    (tl, tr, br, bl) = pts
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


