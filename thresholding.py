import cv2
from PIL import Image
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.filters import threshold_local
import numpy as np


def getThreshold(img):
    if img.dtype != 'uint8':
        img = np.array(255 * img).astype('uint8')

    max_level = np.amax(img)
    hist, grey_levels = histogram(img)
    multiplied = np.multiply(grey_levels, hist)
    summed = np.sum(multiplied)
    pixels_num = img.shape[0] * img.shape[1]
    Tinit = round(summed / pixels_num)
    same = 0
    while not same:
        multiplied = np.multiply(
            grey_levels[grey_levels < Tinit], hist[grey_levels < Tinit])
        summed1 = np.sum(multiplied)
        pixels_num = np.sum(hist[grey_levels < Tinit])
        T1 = round(summed1 / pixels_num)

        multiplied = np.multiply(
            grey_levels[grey_levels >= Tinit], hist[grey_levels >= Tinit])
        summed1 = np.sum(multiplied)
        pixels_num = np.sum(hist[grey_levels >= Tinit])
        T2 = round(summed1 / pixels_num)

        if round((T1 + T2) / 2) == Tinit:
            same = 1
        else:
            Tinit = round((T1 + T2) / 2)
    img[img <= Tinit] = 0
    img[img > Tinit] = 255
    return img


def localThresh(image):
    h, w = image.shape
    image11 = image[0:int(h/2), 0:int(w/2)]
    image1 = getThreshold(image11)

    image12 = image[int(h/2):int(h), 0:int(w/2)]
    image2 = getThreshold(image12)

    image13 = image[0:int(h/2), int(w/2):int(w)]
    image3 = getThreshold(image13)

    image14 = image[int(h/2):int(h), int(w/2):int(w)]
    image4 = getThreshold(image14)
    img1 = getThreshold(image)

    img2 = np.ones_like(image)
    img2[0:int(h/2), 0:int(w/2)] = image1
    img2[int(h/2):int(h), 0:int(w/2)] = image2
    img2[0:int(h/2), int(w/2):int(w)] = image3
    img2[int(h/2):int(h), int(w/2):int(w)] = image4
    return img2


def autsoThreshold(original_image):
    original_image = cv2.GaussianBlur(original_image, (5, 5), 0)
    # Set total number of bins in the histogram
    bins_num = 256
    # Get the image histogram
    hist, bin_edges = np.histogram(original_image, bins=bins_num)
    # Get normalized histogram if it is required
    hist = np.divide(hist.ravel(), hist.max())
    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * \
        weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    threshold = bin_mids[:-1][index_of_max_val]
    #binary = original_image > threshold
    return threshold


def Thresholding_fianl(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY)
    return thresh1


def Thresholding_bradly(input_img):
    #input_img = cv2.resize(input_img, (400, 400))
    h, w = input_img.shape

    S = w / 4
    s2 = S / 2
    T = 15.0

    # integral img
    int_img = np.zeros_like(input_img, dtype=np.uint32)
    for col in range(w):
        for row in range(h):
            int_img[row, col] = input_img[0:row, 0:col].sum()

    # output img
    out_img = np.zeros_like(input_img)

    for col in range(w):
        for row in range(h):
            # SxS region
            y0 = round(max(row - s2, 0))
            y1 = round(min(row + s2, h - 1))
            x0 = round(max(col - s2, 0))
            x1 = round(min(col + s2, w - 1))

            count = (y1 - y0) * (x1 - x0)

            sum_ = int(int_img[y1, x1]) - int(int_img[y0, x1]) - \
                int(int_img[y1, x0]) + int(int_img[y0, x0])

            if input_img[row, col] * count < sum_*(100.-T) / 100.:
                out_img[row, col] = 0
            else:
                out_img[row, col] = 255

    return out_img


def AdaptiveThreshold(gray_image):
    gray_image = cv2.resize(gray_image, (256, 196))
    w, h = gray_image.shape
    S = w / 8
    T = 15.0
    intImg = np.zeros_like(gray_image, dtype=np.uint32)
    for i in range(w):
        sum = 0
        for j in range(h):
            sum = sum + gray_image[i, j]
            if i == 0:
                intImg[i, j] = sum
            else:
                intImg[i, j] = intImg[i-1, j] + sum

    out_img = np.zeros_like(gray_image)
    for i in range(w):
        for j in range(h):
            x1 = i - S//2
            x2 = i + S//2
            y1 = j - S//2
            y2 = j + S//2
            count = (x2-x1) * (y2-y1)
            sum = intImg[x2, y2] - intImg[x2, y1-1] - \
                intImg[x1-1, y2]+intImg[x1-1, y1-1]
            if (gray_image[i, j] * count) <= (sum*(100-T)/100):
                out_img[i, j] = 0
            else:
                out_img[i, j] = 255

    return out_img
#####################################
##########################################
###############################################


def bradley_roth_numpy(image, s=None, t=None):

    # Convert image to numpy array
    img = np.array(image).astype(np.float)

    # Default window size is round(cols/8)
    if s is None:
        s = np.round(img.shape[1]/8)

    # Default threshold is 15% of the total
    # area in the window
    if t is None:
        t = 15.0

    # Compute integral image
    intImage = np.cumsum(np.cumsum(img, axis=1), axis=0)

    # Define grid of points
    (rows, cols) = img.shape[:2]
    (X, Y) = np.meshgrid(np.arange(cols), np.arange(rows))

    # Make into 1D grid of coordinates for easier access
    X = X.ravel()
    Y = Y.ravel()

    # Ensure s is even so that we are able to index into the image
    # properly
    s = s + np.mod(s, 2)

    # Access the four corners of each neighbourhood
    x1 = X - s/2
    x2 = X + s/2
    y1 = Y - s/2
    y2 = Y + s/2

    # Ensure no coordinates are out of bounds
    x1[x1 < 0] = 0
    x2[x2 >= cols] = cols-1
    y1[y1 < 0] = 0
    y2[y2 >= rows] = rows-1

    # Ensures coordinates are integer
    x1 = x1.astype(np.int)
    x2 = x2.astype(np.int)
    y1 = y1.astype(np.int)
    y2 = y2.astype(np.int)

    # Count how many pixels are in each neighbourhood
    count = (x2 - x1) * (y2 - y1)

    # Compute the row and column coordinates to access
    # each corner of the neighbourhood for the integral image
    f1_x = x2
    f1_y = y2
    f2_x = x2
    f2_y = y1 - 1
    f2_y[f2_y < 0] = 0
    f3_x = x1-1
    f3_x[f3_x < 0] = 0
    f3_y = y2
    f4_x = f3_x
    f4_y = f2_y

    # Compute areas of each window
    sums = intImage[f1_y, f1_x] - intImage[f2_y, f2_x] - \
        intImage[f3_y, f3_x] + intImage[f4_y, f4_x]

    # Compute thresholded image and reshape into a 2D grid
    out = np.ones(rows*cols, dtype=np.bool)
    out[img.ravel()*count <= sums*(100.0 - t)/100.0] = False

    # Also convert back to uint8
    out = 255*np.reshape(out, (rows, cols)).astype(np.uint8)

    # Return PIL image back to user
    return Image.fromarray(out)


def thresholding(Rot_image):

    rot_thresholded = bradley_roth_numpy(rgb2gray(Rot_image))
    return rot_thresholded
