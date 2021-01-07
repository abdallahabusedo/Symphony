import cv2
import numpy as np
import skimage
from skimage import io
from skimage.measure import find_contours
from skimage.transform import hough_line
from thresholding import *
import imutils


def getAngle(img):
    # Convert the image to gray-scale
    gray = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray, 50, 200)
    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=10, maxLineGap=250)
    # Draw lines on the image
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        dx = x2 - x1
        dy = y2 - y1
        angles.append(np.rad2deg(np.math.atan2(dy, dx)))

    angles = [angle for angle in angles if angle >= 0]
    if (len(angles) == 0):
        return 0
    average_angle = sum(angles) / len(angles)
    return average_angle


def rotate_image(mat, angle):
    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=[255, 255, 255, 0])
    return rotated_mat


# the first call
def our_rotate(RGBImage):
    angle = getAngle(RGBImage)
    rot = rotate_image(RGBImage, angle)
    return rot


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = np.sum(pts, axis=1)
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


def getFourCorners(image):
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], np.uint8)
    imgContours = image.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = image.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    dilation = cv2.dilate(255 - image, kernel, iterations=5)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 1)

    areas_of_contours = []
    for s in contours:
        areas_of_contours.append(cv2.contourArea(s))

    averageArea = np.sum(areas_of_contours) / (len(areas_of_contours))

    areas = []
    new_contours = []
    for i in range(len(areas_of_contours)):
        if (areas_of_contours[i] >= averageArea):
            areas.append(areas_of_contours[i])
            new_contours.append(contours[i])

    cntrs = new_contours[0]
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    cntr = cntrs[0]

    # draw contour on copy of img as result


    # limit contour to quadrilateral
    peri = cv2.arcLength(cntr, True)
    corners = cv2.approxPolyDP(cntr, 0.04 * peri, True)

    # draw quadrilateral on input image from detected corners
    results = image.copy()
    cv2.polylines(results, [corners], True, (0, 0, 255), 1, cv2.LINE_AA)

    # results = []
    # for c in new_contours:
    #     ll, ur = np.min(c, 0), np.max(c, 0)  # getting the two points
    #     wh = ur - ll  # getting the width and the height
    #     (x, y, w, h) = ll[0][0], ll[0][1], wh[0][1], wh[0][0]
    #     results.append((x, y, w, h))
    # parts = []
    # for box in results:
    #     X, Y, height, width = box
    #     #parts.append(box)
    #     cv2.rectangle(imgBigContour, (int(X), int(Y)), (int(X + width), int(Y + height)), (0, 255, 0), 3)
    #
    print(results)

    return results


def find_corners(image):
    cnt = cv2.findContours(np.uint8(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = cv2.approxPolyDP(cnt[0], 5, True)
    return cnt.astype(np.float32)


def trans(image):
    shapeP = find_corners(image)
    res = four_point_transform(image, shapeP)
    return res
    # shapeRECT =getFourCorners(image)
