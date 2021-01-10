from thresholding import *



def objectDetection(LineRemoved):
    contours, hire = cv2.findContours(255-LineRemoved, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = []
    for j in range(0, len(contours)):
        if hire[0, j, 3] == -1:
            cont.append(contours[j])
    results = []
    for c in cont:
        ll, ur = np.min(c, 0), np.max(c, 0)  # getting the two points
        wh = ur - ll  # getting the width and the height
        (x, y, h, w) = ll[0][0], ll[0][1], wh[0][1], wh[0][0]
        if w*h > 10:  #Habda men el level el te2eel
            results.append((x-3, y-3, w+6, h+6))
    return LineRemoved, results

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
    if(len(angles) == 0):
        return 0
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
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=[255,255,255,0])
    return rotated_mat

# the first call
def our_rotate(RGBImage):
    angle = getAngle(RGBImage)
    rot = rotate_image(RGBImage,angle)
    return rot

