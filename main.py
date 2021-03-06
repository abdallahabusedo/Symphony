import argparse
from transformation import *
from thresholding import *
from preprocessing import *
from to_code import *
import time
import os


##############################
def get_fname_images_tuple(directory):
    fnames = os.listdir(directory)
    to_return = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = io.imread(path)
        to_return.append((fn, gray_scale_image))
    return to_return


def main(in_name, out_name):
    start = time.time()
    images = get_fname_images_tuple(in_name)
    for image in images:
        fn = image[0]
        Original_image = image[1]
        # Original_image = io.imread("inputdata/inputImage/03.PNG")
        # cv2.imshow("original", Original_image)
        # cv2.waitKey()
        ##to modify the light in the image
        b = -30.  # brightness
        c = 190.  # contrast
        Original_image = cv2.addWeighted(np.uint8(Original_image), 1. + c / 127., np.uint8(Original_image), 0, b - c)
        # cv2.imshow("addWeighted", Original_image)
        # cv2.waitKey()
        ################rotating
        Rotate_image = our_rotate(np.asarray(Original_image))
        # cv2.imshow("Rotate_image", Rotate_image)
        # cv2.waitKey()
        #################Thresholding
        thresholed_rotated = thresholding(Rotate_image)
        # thresholed_rotated.save("thres.png")
        removedImages = []
        _, line_positions, Rows_images = divide(np.uint8(thresholed_rotated))
        # cv2.imshow("Rows_images", Rows_images[0])
        # cv2.waitKey()
        biglineArray = []

        for row in Rows_images:
            try:
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
            except:
                continue
        ######### object detection ###############
        objectDetectionImages = []
        finalobject = []
        try:
            ymin, ymax = line_positions[0][0], line_positions[0 + 4][0]
            ymin -= 30
            ymax += 30
        except:
            pass
        i = 0
        k = 0
        num = []
        final_array = []
        final_positions = []
        for j in range(0, len(removedImages)):
            try:
                objectDetectionImg, results = objectDetection(removedImages[j])
                results = sorted(results)
                objectImages = []
                final_positions.append(results)
                for box in results:
                    X, Y, width, height = box
                    if ymin <= Y + (height / 2) <= ymax:
                        cv2.rectangle(objectDetectionImg, (int(X), int(Y)), (int(X + width), int(Y + height)), (0, 0, 0), 1)
                        if width < 0:
                            width = abs(width)
                        if height < 0:
                            height = abs(height)
                        if Y < 0:
                            Y = abs(Y)
                        if X < 0:
                            X = abs(X)
                        symbol = objectDetectionImg[int(Y):int(Y + height), int(X):int(X + width)].copy()
                        symbolline = Rows_images[j][int(Y):int(Y + height), int(X):int(X + width)].copy()
                        # cv2.imshow("haha", symbolline)
                        # cv2.waitKey(0)
                        if symbol.shape[0] != 0 and symbol.shape[1] != 0:
                            finalobject.append(box)
                            objectImages.append(symbol)
                            # objectImageswithlines.append(symbolline)
                            # Image.fromarray(symbol).save(out_path + str(i) + ".png")
                            # Image.fromarray(symbolline).save(out_path + str(i + 1) + ".png")
                        i += 2
                final_array.append(objectImages)
                num.append(len(finalobject))
                objectDetectionImages.append(objectDetectionImg)
            except:
                continue
        rowIndex = 0
        finallllllll = []
        for i in range(len(final_array)):
            try:
                code_line = ['[']
                ob = final_array[i]
                objectIndex = 0
                asda = biglineArray[i]
                for xc in ob:
                    path = get_similar_temp(xc)
                    code = to_code(final_positions[i][objectIndex][1],
                                   final_positions[i][objectIndex][1] + final_positions[i][objectIndex][3],
                                   asda, path)
                    code_line.append(code)
                    objectIndex += 1
                code_line.append(']')
                code_string = arrange_code_string(code_line)
                finallllllll.append(code_string)
            except:
                continue
        finallllllll = "\n".join(finallllllll)
        file1 = open(r"" + out_name + "/" + fn[:fn.index(".")] + ".txt", "a")
        file1.write(finallllllll)
    file1.close()
    end = time.time()
    print(f"Runtime of the program is {end - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case_file_path", help="Test Case")
    parser.add_argument("output_file_path", help="Output File Name")
    args = parser.parse_args()
    in_name = args.test_case_file_path
    out_name = args.output_file_path
    main(in_name, out_name)
