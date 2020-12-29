from skimage.filters import threshold_local

def Local_Thresholding(original_image):
    block_size = 51
    # calculate the local threshold value
    threshold_local_value = threshold_local(original_image, block_size, offset=0.0001)
    # apply the local threshold value on the image
    binary = original_image > threshold_local_value
    return binary

# general
# otsu