
















image = io.imread('26.jpg',as_gray="true")
binary_image = Local_Thresholding(image)
res = getFourCorners(binary_image)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#result = divide(gray)
#result3 = lineRemover(result[0])
#result = objectDetection(result)
io.imshow(res)
io.show()