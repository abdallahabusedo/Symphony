
#the main for one image
from skimage.color import rgb2gray
from transformation import *
from thresholding import *
from ReadWrite import *
##############################
Original_image = io.imread('inputdata/test 2/07.png')

################rotating
Rotate_image = our_rotate(Original_image)

#################Thresholding
thresholed_rotated = thresholding(Rotate_image)

############################
##############prespect


io.imshow(np.uint8(thresholed_rotated))
io.show()


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