import numpy as np
import cv2
import skimage
from skimage.measure import find_contours
from skimage import io
import scipy.ndimage
import matplotlib.pyplot as plt
from math import ceil
from skimage.color import rgb2gray
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin

####get staff lines by horizontal projections
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
#-----------------------------------------------------------------------------------------------------------------------------------------
#this function divides the music sheet to sub images each containing a row of the notes 
#-----------------------------------------------------------------------------------------------------------------------------------------
def divide(output_path,img):
    kernel =  np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ],np.uint8)
    kernel2 =  np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ],np.uint8)
    #---------------------------dilates the image to get the staff lines only-------------------------------------------------------------
    retval, img_binary = cv2.threshold(img,215,255,type=cv2.THRESH_BINARY)
    dilation = cv2.dilate(img_binary,kernel,iterations =50)
    Img_h,Img_w=img.shape
#---------------------------finds the contoours that surround the lines ------------------------------------------------------------
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 50)) #the structure element
    img_closing = binary_closing(dilation,rect)
    contours = find_contours(dilation,0.8)
    results = []

    for c in contours :
        ll, ur = np.min(c, 0), np.max(c, 0) #getting the two points
        wh = ur - ll  #getting the width and the height
        (x,y,w,h) = ll[0], ll[1], wh[1], wh[0]
        results.append((x,y,w,h)) #getting the 4 contours that we have (4 groups of the numbers)
    #When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
    line_positions=results
    for box in results:
        X, Y, width, height = box
        cv2.rectangle(dilation, (int(Y), int(X)), (int(Y+width), int(X+height)), (0, 255, 0),10)
#---------------------------finds the main big contours to cut the image-------------------------------------------------------------
    img_closing = binary_closing(dilation,rect)
    contours = find_contours(dilation,0.8)
    results = []
    for c in contours :
        ll, ur = np.min(c, 0), np.max(c, 0) #getting the two points
        wh = ur - ll  #getting the width and the height
        (x,y,w,h) = ll[0], ll[1], wh[1], wh[0]
        results.append((x,y,w,h)) #getting the 4 contours that we have (4 groups of the numbers)
    #When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
    i=1
    xup=0
    l=len(results)
    for box in results:
        X, Y, width, height = box
        if i == l:
            xl=Img_h
        else:
            xl,yl,widthl,heightl=results[i]
        cv2.rectangle(dilation, (int(Y), int(X)), (int(Y+width), int(X+height)), (0, 255, 0),1)
        Image=img[int(X-(X-xup)/2):int(X+height+((xl-X)/2)),0:int(Img_w),] #Y-50
        cv2.imwrite((output_path+str(i)+".bmp"),Image)
        i=i+1
        xup=X+height
    return l ,line_positions
#------------------------------------------------------------------------------------------------------------
    ####### removing staff lines 
    ## for more clear img ,uncomment binarization
def remove_lines(out_path,in_path,img_count):
    kernel=np.array([[0, 0, 0],[1, 1, 1],[0, 0, 0]],np.uint8)
    kernel2 =  np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ],np.uint8)
    for i in range(1,img_count):
        img = cv2.imread((in_path+str(i)+".bmp"))
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        erosion = cv2.erode(gray,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel2,iterations =1)
        cv2.imwrite((out_path+str(i)+".bmp"),dilation)
#----------------------------------------------------------------------------------------------------------------
in_path="D:\\college\\cmp3\\first\\ip\\symphony\\Symphony\\note3.jpeg"
division_out_path="D:\\college\\cmp3\\first\\ip\\symphony\\Symphony\\cut"
lines_removed_out_path="D:\\college\\cmp3\\first\\ip\\symphony\\Symphony\\lines_removed"
img = cv2.imread(in_path,0)
img_count,line_positions=divide(division_out_path,img) #divide the image to small images containing each row
remove_lines(lines_removed_out_path,division_out_path,img_count)
