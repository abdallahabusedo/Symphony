from collections import Counter
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import filters
from skimage import io
import skimage
from PIL import Image as ii
# returns an array of the number of black pixels in each row of the picture
from skimage.measure import find_contours
from skimage.morphology import binary_closing


def horizontalProjection(bwArray):
    rows = len(bwArray)
    cols = len(bwArray[0])
    pixCount = [0] * rows
    for row in range(rows):
        for col in range(cols):
            if not bwArray[row][col]:
                pixCount[row] += 1
    return pixCount


# finds the rows that have black pixels along most of the picture
def getLines(pixArray, picWidth):
    lineArray = []
    for row in range(len(pixArray)):
        if pixArray[row] >= (picWidth * 0.8):  # take only the lines with specific length
            lineArray.append(row)  # append the row that the line is found in
    return lineArray


def findBarLineWidth(lineArray):
    lineThicknesses = []
    newLineArray = []
    i = 0
    thickness = 1  # first line is always a barline
    while i < len(lineArray) - 1:
        if thickness == 1:
            newLineArray.append(lineArray[i])

        if lineArray[i] + 1 == lineArray[i + 1]:
            thickness += 1
        else:
            lineThicknesses.append(thickness)
            thickness = 1
        i += 1
    lineThicknesses.append(thickness + 1)
    if all(val == lineThicknesses[0] for val in lineThicknesses):
        return [lineThicknesses[0], newLineArray]
    return [lineThicknesses, newLineArray]


def findSpacesSize(lineArray, lineThickness):
    lineDistances = []
    for i in range(len(lineArray) - 1):
        lineDistances.append(lineArray[i + 1] - lineArray[i])
    # gets the mode of the array(most common space size)
    spacesCount = Counter(lineDistances)
    spaceBetweenBars = max(lineDistances)
    tempSpaceInfo = spacesCount.most_common(1)  # returns the most frequent distance
    commonSize = tempSpaceInfo[0][0]
    if commonSize == 1:  # fixing the commonSize
        j = 0
        while j < len(lineDistances):
            if lineDistances[j] != 1:
                tempCommon = lineDistances[j]
                if lineDistances[j + lineThickness[0]] > (tempCommon - 1) * 0.9 and lineDistances[
                    j + lineThickness[0]] < (tempCommon + 1) * 1.1:
                    commonSize = tempCommon
                    break
            j += 1
    i = 0
    count = 0
    spaceSizeArr = []
    while i < len(lineDistances):
        # space sizes can be inconsistent, so if its within ~10% then its accepted
        if lineDistances[i] > (commonSize - 1) * 0.9 and lineDistances[i] < (commonSize + 1) * 1.1:
            spaceSizeArr.append(lineDistances[i])
            count += 1
        else:
            if lineDistances[i] != 1:  # if its 1 then its part of the same line so dont reset
                count = 0
                spaceSizeArr = []
        if count == 4:
            return [spaceSizeArr, spaceBetweenBars]
        i += 1
    return []


# removes the bar lines from the sheet music
def removeBarLines(lineLocations, pixels, barLineWidth, picWidth):
    oneLineThickness = (len(barLineWidth) == 1)
    lineThickness = barLineWidth[0]
    lineNum = 0
    lineCounter = 0
    while lineNum < len(lineLocations):
        if not oneLineThickness:
            lineThickness = barLineWidth[lineCounter]
        pixels = eraseLine(lineThickness, lineLocations[lineNum], pixels, picWidth)
        lineNum += 1
        lineCounter += 1
    return pixels


# removes a single line with given thickness from the sheet music without
# effecting any other objects on the sheet
def eraseLine(thickness, startLine, image, picWidth):  # doesn't erase line, but make them white
    topLine = startLine
    botLine = startLine + thickness - 1
    for col in range(picWidth):
        if image.item(topLine, col) == 0 or image.item(botLine, col) == 0:
            if image.item(topLine - 1, col) == 255 and image.item(botLine + 1, col) == 255:
                for j in range(thickness):
                    image.itemset((topLine + j, col), 255)
            elif image.item(topLine - 1, col) == 255 and image.item(botLine + 1, col) == 0:
                thick = thickness + 1
                if thick < 1:
                    thick = 1
                for j in range(int(thick)):
                    image.itemset((topLine + j, col), 255)

            elif image.item(topLine - 1, col) == 0 and image.item(botLine + 1, col) == 255:
                thick = thickness + 1
                if thick < 1:
                    thick = 1
                for j in range(int(thick)):
                    image.itemset((botLine - j, col), 255)
    return image


def removeMe(bwImg, lineLocations, barLineWidth):
    pixels = np.array(bwImg)  # pixels(row, col)
    picWidth = len(np.asarray(bwImg)[0])  # bwImg.size[0]
    pixels = removeBarLines(lineLocations, pixels, barLineWidth, picWidth)
    return pixels


def divide(img):
        kernel = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ], np.uint8)
        kernel2 = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ], np.uint8)
        # ---------------------------dilates the image to get the staff lines only-------------------------------------------------------------
        # retval, img_binary = cv.threshold(img, 215, 255, type=cv.THRESH_BINARY)
        dilation = cv.dilate(img, kernel, iterations=50)
        ii.fromarray(dilation).save('out.png')

        Img_h, Img_w = img.shape
        # ---------------------------finds the contours that surround the lines ------------------------------------------------------------
        rect = cv.getStructuringElement(cv.MORPH_RECT, (60, 50))  # the structure element
        contours = find_contours(dilation, 0.8)
        results = []
        width_array = []
        height_array = []
        for c in contours:
            ll, ur = np.min(c, 0), np.max(c, 0)  # getting the two points
            wh = ur - ll  # getting the width and the height
            (x, y, w, h) = ll[0], ll[1], wh[1], wh[0]
            results.append((x, y, w, h))
            width_array.append(w)
            height_array.append(h)
        hi = abs(results[0][0]-results[1][0])
        hi= hi*2
        print(hi,results[0][0],results[1][0])
        widthCount = Counter(width_array)
        commonWidth = widthCount.most_common(1)  # returns the most frequent width

        HeightCount = Counter(height_array)
        commonHeight = HeightCount.most_common(1)  # returns the most frequent height

        new_results = []
        for result in results:
            if result[2] >= commonWidth[0][0] * 0.8 and result[2] <= commonWidth[0][0] * 1.2:
                new_results.append(result)

        new_new_results = []

        for result in new_results:
            if result[3] >= commonHeight[0][0] * 3:
                new_new_results.append((result[0] + result[3], result[1], result[2], commonHeight[0][0]))
            else:
                new_new_results.append((result[0], result[1], result[2], result[3]))

        # new_new_result -> has proper widths and heights for lines

        # When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
        line_positions = new_new_results

        for box in new_new_results:
            X, Y, width, height = box
            cv.rectangle(dilation, (int(Y), int(X)), (int(Y + width), int(X + height)), (0, 255, 0), int(hi))

        # ---------------------------finds the main big contours to cut the image-------------------------------------------------------------

        contours = find_contours(dilation, 0.8)
        results = []
        for c in contours:
            ll, ur = np.min(c, 0), np.max(c, 0)  # getting the two points
            wh = ur - ll  # getting the width and the height
            (x, y, w, h) = ll[0], ll[1], wh[1], wh[0]
            # getting the 4 contours that we have (4 groups of the numbers)
            results.append((x, y, w, h))
        # When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
        i = 1
        xup = 0
        l = len(results)
        ROWSImages = []
        for box in results:
            X, Y, width, height = box
            if i == l:
                xl = Img_h
            else:
                xl, yl, widthl, heightl = results[i]
            cv.rectangle(dilation, (int(Y), int(X)), (int(Y + Img_w), int(X + height)), (0, 255, 0), 1)
            Image = img[int(X - (X - xup) / 2):int(X + height + ((xl - X) / 2)), 0:int(Img_w), ]  # Y-50
            ROWSImages.append(Image)
            i = i + 1
            xup = X + height
        return l, line_positions, ROWSImages
