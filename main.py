# the main for one image
from networkx.drawing.tests.test_pylab import plt
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
from skimage.morphology import dilation, closing, erosion, remove_small_objects, disk
from skimage import io
from transformation import *
from thresholding import *
from ReadWrite import *
from preprocessing import *
from to_code import *
import time

##############################
start = time.time()

Original_image = io.imread(r'inputdata/test 2/02.PNG')

##to modify the light in the image
b = -30.  # brightness
c = 190.  # contrast
Original_image = cv2.addWeighted(np.uint8(Original_image), 1. + c / 127., np.uint8(Original_image), 0, b - c)

################rotating
Rotate_image = our_rotate(np.asarray(Original_image))
#################Thresholding
thresholed_rotated = thresholding(Rotate_image)
_, line_positions, Rows_images = divide(np.uint8(thresholed_rotated))

removedImages = []
biglineArray = []
for row in Rows_images:
    bwArray = np.array(row)
    picWidth = len(bwArray[0])
    horzPicCount = horizontalProjection(bwArray)
    lineArray = getLines(horzPicCount, picWidth)
    lineThickness, newLineArray = findBarLineWidth(lineArray)
    lineArray = newLineArray
    biglineArray.append(lineArray)
    spaceSize, spaceBetweenBars = findSpacesSize(lineArray, lineThickness)
    removed_line_pic = removeMe(row, lineArray, lineThickness)
    removedImages.append(removed_line_pic)

######### object detection ###############
objectDetectionImages = []

finalobject = []
ymin, ymax = line_positions[0][0], line_positions[0 + 4][0]
ymin -= 30
ymax += 30
out_path = "datasets/"
i = 0
k = 0
objectImages = []
num = []
objectImageswithlines = []
# to be sent to shaza w areej
final_array = []
final_positions = []
for j in range(0, len(removedImages)):
    objectDetectionImg, results = objectDetection(removedImages[j])
    results = sorted(results)
    objectImages = []
    print(results)
    final_positions.append(results)
    for box in results:
        X, Y, width, height = box
        if ymin <= Y + (height / 2) <= ymax:
            cv2.rectangle(objectDetectionImg, (int(X), int(Y)), (int(X + width), int(Y + height)), (0, 255, 0), 1)
            symbol = objectDetectionImg[int(Y):int(Y + height), int(X):int(X + width)].copy()
            symbolline = Rows_images[j][int(Y):int(Y + height), int(X):int(X + width)].copy()
            if symbol.shape[0] != 0 and symbol.shape[1] != 0:
                finalobject.append(box)
                objectImages.append(symbol)
                objectImageswithlines.append(symbolline)
                Image.fromarray(symbol).save(out_path + str(i) + ".png")
                Image.fromarray(symbolline).save(out_path + str(i + 1) + ".png")
            i += 2
    final_array.append(objectImages)
    num.append(len(finalobject))
    objectDetectionImages.append(objectDetectionImg)

rowIndex = 0
finallllllll = []
for i in range(len(final_array)):  # 0 -> 3
    code_line = ['[']
    ob = final_array[i]
    #print("ob", len(ob))
    objectIndex = 0
    for xc in ob:
        #print("Asd",final_positions[objectIndex][0],final_positions[objectIndex][0] + final_positions[objectIndex][3] ,biglineArray[i])
        path = get_similar_temp(xc)
        #print("print" , path)
        code = to_code(final_positions[i][objectIndex][0],
                       final_positions[i][objectIndex][0] + final_positions[i][objectIndex][3],
                       biglineArray[i], path)
        code_line.append(code)
        objectIndex += 1
    code_line.append(']')
    code_string = arrange_code_string(code)
    finallllllll.append(code_string)
finallllllll = "\n".join(finallllllll)

print(finallllllll)
end = time.time()
print(f"Runtime of the program is {end - start}")
