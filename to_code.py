import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from skimage import io
import glob
import itertools

forlders_guide = {'a_1': ' a/1', 'a_2': ' a/2', 'a_4': ' a/4', ' a_8': ' a/8', 'a_8d_8': ' a/8 d/8',
                  'a_16': ' a/16', 'a_32': ' a/32', 'a1_8b1_8': ' a1/8 b1/8', 'b_2': ' b/2', 'b_4': ' b/4', 'clef': ' ',
                  'c1_8g1_8': ' c1/8 g1/8', 'cdef1_16': ' c1/16 d1/16 e1/16 f1/16', ' cgde1_16': ' c1/16 g1/16 d1/16 e1/16',
                  'cgde2_16': ' c2/16 g2/16 d2/16 e2/16', '4e_16': ' e1/16 e1/16 e1/16 e1/16', 'e_8c_8': ' e/8 c/8',
                  'fgab1_16': ' f1/16 g1/16 a1/16 b1/16', 'b14_e14_g14': ' {b/14 e/14 g/14}',
                  'b14_f14_g14': ' {b/14 f/14 g/14}',
                  'c14_e14_g14': ' {c/14 e/14 g/14}', 'g14_e14': ' {g/14 e/14}', '#': ' #', '##': ' ##', '&': ' &',
                  '&&': ' &&', '.hi': '.',
                  '2': '2', '4': '4'
                  }


def arrange_code_string(code_line):
    i = -1
    for c in code_line:
        i += 1
        if i == 2:
            if c == '4':
                code_line[2] = code_line[2].replace('4', 'meter <"4/')
                continue
            elif c == '2':
                code_line[2] = code_line[2].replace('2', 'meter <"2/')
                continue
        elif i == 3:
            if c == '4':
                code_line[3] = code_line[3].replace('4', '4">')
                continue
            elif c == '2':
                code_line[3] = code_line[3].replace('2', '2">')
                continue
        if c == '#' or c == '##' or c == '&' or c == '&&':
            symbol = c
            code_line[i] = ' '
            code_old = code_line[i + 1]
            # code_new = code_old.split('/')[0] + symbol + '/' + code_old.split('/')[1]
            code_new = code_old[0]+symbol+code_old[1:len(code_old)]
            code_line[i + 1] = code_new
    code_string = "".join(code_line)
    return code_string


def to_code(start, end, lines, folder_name):
    code = forlders_guide.get(folder_name)
    space = lines[1] - lines[0]
    if code == 'a/1' or code == 'a/2' or code == 'a/4' or code == 'a/8' or code == 'a/16' or code == 'a/32':
        if   end > lines[4]+space and lines[2] > start > lines[1]:
            code = code.replace('a', 'c1')
        elif  lines[4]+0.7*space < end  and  start < lines[1] :
            code = code.replace('a', 'd1')
        elif  lines[4] < end  and  lines[0] < start < lines[1] :
            code = code.replace('a', 'e1')
        elif  lines[2] < end < lines[3]+ 0.3*space  and  lines[0] > start :
            code = code.replace('a', 'a1')
        elif  lines[3] < end < lines[4]  and  lines[0] > start :
            code = code.replace('a', 'g1')
        else :
            code = code.replace('a', 'f1')


    # --------------------------------------------------------------------------------------------------
    if code == 'b/1' or code == 'b/2' or code == 'b/4' or code == 'b/8' or code == 'b/16' or code == 'b/32':
        if lines[0] - 2 * space >= start >= lines[0] - 2.5 * space :
            code = code.replace('b', 'b2')
        elif lines[0] - 1.5 * space >= start >= lines[0] - 2 * space:
            code = code.replace('b', 'a2')
        elif lines[3] > start > lines[1] and end > lines[4]:
            code = code.replace('b', 'b1')
        elif lines[2] >= start >= lines[1]:
            code = code.replace('b', 'c2')
        elif lines[1] >= start >= lines[0] and lines[3] <= end <=lines[4] + 0.3*space:
            code = code.replace('b', 'e2')
        elif lines[1] > start >  lines[0]-0.5*space:
            code = code.replace('b', 'd2')
        elif lines[0] > start > lines[0]-0.7*space and end > lines[3] + 0.5*space:
            code = code.replace('b', 'f2')
        elif lines[0] >= start >= lines[0] - 1.3* space:
            code = code.replace('b', 'g2')
    return code


def get_similar_temp(template):
    folders = glob.glob("ourdata/*")
    imagenames_list = []  # path to img
    folder_name = []
    for folder in folders:
        for f in glob.glob(folder + '/*'):
            imagenames_list.append(f)
            g = f.split('\\')
            l = len(g)
            folder_name.append(g[l - 2])
            # print(g[l - 2])
            # print (g)
    read_images = []
    for image in imagenames_list:
        read_images.append(io.imread(image))
    max_sim = 0
    max_path = ""
    max_img = []
    max_loc = 0
    for i, j in zip(read_images, folder_name):
        # print(j)
        # print(i.shape)
        if len(i.shape) != 2:
            img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        else:
            img = i
        h, w = template.shape[::]
        if h <= img.shape[0] and w <= img.shape[1]:
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            plt.imshow(res, cmap='gray')
            maxx = []
            for r in res:
                maxx.append(max(r))
            # print(max(maxx) * 100)
            if (max(maxx) * 100 >= max_sim):
                max_sim = max(maxx) * 100
                g = j.split('\\')
                l = len(g)
                max_path = g[l - 2]
                max_img = i
                max_loc = np.where(res == max(maxx))
            loc = np.where(res == max(maxx))
            # print(loc)
            # Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.
            # Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
            # then the second item and then third, etc.
    #     else:
    #         # print("no")
    # print("max_sim: ", max_sim)
    # print("path: ", max_path)
    # print("max_loc: ", max_loc)
    for pt in zip(*max_loc[::-1]):  # -1 to swap the values as we assign x and y coordinate to draw the rectangle.
        cv2.rectangle(max_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 1)
        # cv2.imshow("Matched image", max_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    return max_path
