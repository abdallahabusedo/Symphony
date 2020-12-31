import cv2
import numpy as np
import skimage
from skimage import io
from skimage.transform import hough_line
from thresholding import *
import imutils



def getAngle(img):
    # Convert the image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    average_angle = sum(angles) / len(angles)
    return average_angle

def rotate_image(mat, angle):
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

# the first call
def our_rotate(RGBImage):
    angle = getAngle(RGBImage)
    rot = rotate_image(RGBImage,angle)
    return rot

