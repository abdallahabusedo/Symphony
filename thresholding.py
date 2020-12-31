import skimage
from skimage import io
from skimage.exposure import histogram
from skimage.filters import threshold_local, threshold_otsu, threshold_yen
import numpy as np

def Local_Thresholding(original_image):
    block_size = 31
    # calculate the local threshold value
    threshold_local_value = threshold_local(original_image, block_size, offset=1)
    # apply the local threshold value on the image
    binary = original_image > threshold_local_value
    return binary

# general
# otsu

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
        multiplied = np.multiply(grey_levels[grey_levels < Tinit], hist[grey_levels < Tinit])
        summed1 = np.sum(multiplied)
        pixels_num = np.sum(hist[grey_levels < Tinit])
        T1 = round(summed1 / pixels_num)

        multiplied = np.multiply(grey_levels[grey_levels >= Tinit], hist[grey_levels >= Tinit])
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
    h,w=image.shape
    image11=image[0:int(h/2),0:int(w/2)]
    image1=getThreshold(image11)

    image12=image[int(h/2):int(h),0:int(w/2)]
    image2=getThreshold(image12)

    image13=image[0:int(h/2),int(w/2):int(w)]
    image3=getThreshold(image13)

    image14=image[int(h/2):int(h),int(w/2):int(w)]
    image4=getThreshold(image14)
    img1=getThreshold(image)

    img2=np.ones_like(image)
    img2[0:int(h/2),0:int(w/2)]=image1
    img2[int(h/2):int(h),0:int(w/2)]=image2
    img2[0:int(h/2),int(w/2):int(w)]=image3
    img2[int(h/2):int(h),int(w/2):int(w)]=image4
    return img2





