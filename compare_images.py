import cv2
import numpy as np
original= cv2.imread("D:/college/cmp3/first/ip/symphony/Symphony/1circleo.jpg")
image_to_compare= cv2.imread("D:/college/cmp3/first/ip/symphony/Symphony/1circle.jpg")

def compare_images(original,image_to_compare):
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None) #finds key points and give every key point value to represent it in the descriptors
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params) #organize descriptors 
    matches = flann.knnMatch(desc_1, desc_2, k=2) #compare descriptors and find matching points
    good_points = []
    ratio = 0.6 #the bigger the more false good points , the smaller the the harder to find good points 
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good_points.append(m)
    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
    result2 = cv2.drawMatchesKnn(original, kp_1, image_to_compare, kp_2, matches, None)
    if len(kp_1) < len(kp_2):
        sim=(len(good_points)/len(kp_1))*100
    else:
        sim=(len(good_points)/len(kp_2))*100



    print(sim)
    cv2.imshow("result", result)
    cv2.imshow("matches", result2)
    cv2.imshow("Original", original)
    cv2.imshow("image_to_compare", image_to_compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return sim