 #  1 (❤ ω ❤)生成中间文件txt 标签+bndbox
import numpy as np
import cv2
import os
import re
import sys
import string

# 在棋盘倾斜的情况下，用四个角的坐标计算整个棋盘19X19的位置坐标
def PointPerspectiveTransform(src, h):
    u = src[0];
    v = src[1];
    x = (h[0][0] * u + h[0][1] * v + h[0][2]) / (h[2][0] * u + h[2][1] * v + h[2][2]);
    y = (h[1][0] * u + h[1][1] * v + h[1][2]) / (h[2][0] * u + h[2][1] * v + h[2][2]);
    return [x, y]
def getBoardGrid(point4, WY_BOARD_LINES):
    boardgrid = np.zeros((361,2))
    src = np.array([[0, 0], [1800, 0], [0, 1800], [1800, 1800]], np.float32)
    dst = np.array([point4[0], point4[1], point4[2], point4[3]],  np.float32)
    h = cv2.getPerspectiveTransform(src, dst)
    for x in range(0, WY_BOARD_LINES):
        for y in range(0, WY_BOARD_LINES):
            src = [x * 100, y * 100]
            des = PointPerspectiveTransform(src, h)
            boardgrid[x+WY_BOARD_LINES*y] = des
    return boardgrid

#返回bndbox的Y集合，Y（20，19）
def getVOCBox(boardgrid):
    bndbox_y = np.zeros([20,19])
    bndbox_x = np.zeros([19,20])
    # Y
    # 1~18行，取中点：
    for i in range(1,19):
        for j in range(19):
            bndbox_y[i, j] =( boardgrid[19*i+j, 1] + boardgrid[19*(i-1)+j, 1]) /2
    # 0、19行，外围
    i = 0
    for j in range(19):
        bndbox_y[i, j] = 2*boardgrid[j, 1] - bndbox_y[1, j]
    i = 19
    for j in range(19):
        bndbox_y[i, j] = 2*boardgrid[342+j, 1] - bndbox_y[18, j]

    # X
    # 1~18列，取中点：
    for j in range(1,19):
        for i in range(19):
            bndbox_x[i, j] =( boardgrid[19*i+j, 0] + boardgrid[19*i+j-1, 0]) /2
    # 0、19列，外围
    j = 0
    for i in range(19):
        bndbox_x[i, j] = 2 * boardgrid[i*19, 0] - bndbox_x[i, 1]
    j = 19
    for i in range(19):
        bndbox_x[i, j] = 2 * boardgrid[19*(i+1)-1, 0] - bndbox_x[i, 18]
    return bndbox_y, bndbox_x

# 361 * 4 (xmin, ymin, xmax, ymax)
def generateBndBox_1_1(bndbox_y , bndbox_x):
    bndbox = np.zeros([361, 4])
    index = 0
    for i in range (19):
        for j in range (19):
            bndbox[index, 0] = bndbox_x[i, j]
            bndbox[index, 1] = bndbox_y[i, j]
            bndbox[index, 2] = bndbox_x[i, j+1]
            bndbox[index, 3] = bndbox_y[i+1, j]
            index = index + 1
    return bndbox
def generateBndBox_corner(boardgrid):
    bndbox = np.zeros([4, 4])
    LONG = 1.6
    # lefttop
    delta_x = boardgrid[1, 0] - boardgrid[0, 0]
    delta_y = boardgrid[19, 1] - boardgrid[0, 1]
    bndbox[0, 0] = boardgrid[0, 0] - LONG*delta_x
    bndbox[0, 1] = boardgrid[0, 1] - LONG*delta_y
    bndbox[0, 2] = boardgrid[0, 0] + LONG*delta_x
    bndbox[0, 3] = boardgrid[0, 1] + LONG*delta_y
    # righttop
    delta_x = boardgrid[18, 0] - boardgrid[17, 0]
    delta_y = boardgrid[37, 1] - boardgrid[18, 1]
    bndbox[1, 0] = boardgrid[18, 0] - LONG*delta_x
    bndbox[1, 1] = boardgrid[18, 1] - LONG*delta_y
    bndbox[1, 2] = boardgrid[18, 0] + LONG*delta_x
    bndbox[1, 3] = boardgrid[18, 1] + LONG*delta_y
    # leftbottom
    delta_x = boardgrid[343, 0] - boardgrid[342, 0]
    delta_y = boardgrid[342, 1] - boardgrid[323, 1]
    bndbox[2, 0] = boardgrid[342, 0] - LONG*delta_x
    bndbox[2, 1] = boardgrid[342, 1] - LONG*delta_y
    bndbox[2, 2] = boardgrid[342, 0] + LONG*delta_x
    bndbox[2, 3] = boardgrid[342, 1] + LONG*delta_y
    # rightbottom
    delta_x = boardgrid[360, 0] - boardgrid[359, 0]
    delta_y = boardgrid[360, 1] - boardgrid[341, 1]
    bndbox[3, 0] = boardgrid[360, 0] - LONG*delta_x
    bndbox[3, 1] = boardgrid[360, 1] - LONG*delta_y
    bndbox[3, 2] = boardgrid[360, 0] + LONG*delta_x
    bndbox[3, 3] = boardgrid[360, 1] + LONG*delta_y
    for i in range(4):
        for j in range(4):
            bndbox[i, j] = int(bndbox[i, j])
    return bndbox

# (xmin, ymin, xmax, ymax)
def generateBndBox_2_1(bndbox_y , bndbox_x):
    bndbox = np.zeros([361, 4])
    index = 0
    for i in range (19-1):
        for j in range (19):
            # print(i,"    ",j)
            bndbox[index, 0] = (bndbox_x[i, j] + bndbox_x[i+1, j]) / 2
            bndbox[index, 1] = bndbox_y[i, j]
            bndbox[index, 2] = (bndbox_x[i, j+1] + bndbox_x[i+1, j+1]) / 2
            bndbox[index, 3] = bndbox_y[i+2, j]
            index = index + 1
    return bndbox

# 306 * 4 (xmin, ymin, xmax, ymax) 1*2的
def generateBndBox_1_2(bndbox_y , bndbox_x):
    bndbox = np.zeros([361, 4])
    index = 0
    for i in range (19):
        for j in range (19-1):
            bndbox[index, 0] = bndbox_x[i, j]
            bndbox[index, 1] = (bndbox_y[i, j] + bndbox_y[i, j+1])/2
            bndbox[index, 2] = bndbox_x[i, j+2]
            bndbox[index, 3] = (bndbox_y[i+1, j] + bndbox_y[i+1, j+1])/2
            index = index + 1
    return bndbox

def add_sort(point4):
    dict = {point4[0][0]: point4[0][1], point4[1][0]: point4[1][1], point4[2][0]: point4[2][1],
            point4[3][0]: point4[3][1]}  # (x,y)
    my_list = [point4[0][0], point4[1][0], point4[2][0], point4[3][0]]
    my_list.sort()
    # print(my_list)
    if dict[my_list[0]] < dict[my_list[1]]:
        point4[0][0] = my_list[0]
        point4[0][1] = dict[my_list[0]]
        point4[2][0] = my_list[1]
        point4[2][1] = dict[my_list[1]]
    else:
        point4[0][0] = my_list[1]
        point4[0][1] = dict[my_list[1]]
        point4[2][0] = my_list[0]
        point4[2][1] = dict[my_list[0]]

    if dict[my_list[2]] < dict[my_list[3]]:
        point4[1][0] = my_list[2]
        point4[1][1] = dict[my_list[2]]
        point4[3][0] = my_list[3]
        point4[3][1] = dict[my_list[3]]
    else:
        point4[1][0] = my_list[3]
        point4[1][1] = dict[my_list[3]]
        point4[3][0] = my_list[2]
        point4[3][1] = dict[my_list[2]]
    return point4

def generate_bndbox_txt_1_1(corner4txt_path, label_filepath, label_loc_path):
    fileList = os.listdir(corner4txt_path)
    os.chdir(corner4txt_path)
    for fileName in fileList:
        ## 获取 18 * 17 个 bndbox
        f_read = corner4txt_path + fileName
        print(f_read)
        with open(f_read, "r") as f:
            a = f.read().split('\t')  # 读取文件
        point4 = np.zeros(8)
        for i in range(len(a)):
            point4[i] = int(a[i])
        point4 = np.reshape(point4, (4, 2))
        point4 = add_sort(point4) # add sort
        boardgrid = getBoardGrid(point4, 19)
        bndbox_y, bndbox_x = getVOCBox(boardgrid)
        bndbox = generateBndBox_1_1(bndbox_y, bndbox_x)

        ## 获取 label
        label_filepath_f = label_filepath + fileName
        all_label = np.zeros(19 * 19) - 1
        index = 0
        with open(label_filepath_f) as f:
            a = f.read()
            for i in range(len(a)):
                if a[i] == '0' or a[i] == '1' or a[i] == '2':
                    all_label[index] = int(a[i])
                    index += 1
            all_label = all_label.reshape(19, 19)

        index2 = 0
        save_f = label_loc_path + fileName
        with open(save_f, "w") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
            for i in range(19):
                for j in range(19):
                    if all_label[i][j] == 0:
                        label = 'empty'
                    elif all_label[i][j] == 1:
                        label = 'white'
                    else:
                        label = 'black'
                    label_bnd = label + ' ' + str(int(bndbox[index2][0])) + ' ' + str(
                        int(bndbox[index2][1])) + ' ' + str(int(bndbox[index2][2])) + ' ' + str(
                        int(bndbox[index2][3])) + '\n'
                    index2 += 1
                    file.write(label_bnd)

def generate_bndbox_txt_1_2(corner4txt_path, label_filepath, label_loc_path):
    fileList = os.listdir(corner4txt_path)
    os.chdir(corner4txt_path)
    for fileName in fileList:
        ## 获取 19 * 18 个 bndbox
        f_read = corner4txt_path + fileName
        print(f_read)
        with open(f_read, "r") as f:
            a = f.read().split('\t')  # 读取文件
        point4 = np.zeros(8)
        for i in range(8):
            point4[i] = int(a[i])
        point4 = np.reshape(point4, (4, 2))
        point4 = add_sort(point4)  # add sort
        boardgrid = getBoardGrid(point4, 19)
        bndbox_y, bndbox_x = getVOCBox(boardgrid)
        bndbox = generateBndBox_1_2(bndbox_y, bndbox_x)

        ## 获取 label
        label_filepath_f = label_filepath + fileName
        all_label = np.zeros(19 * 19) - 1
        index = 0
        with open(label_filepath_f) as f:
            a = f.read()
            for i in range(len(a)):
                if a[i] == '0' or a[i] == '1' or a[i] == '2':
                    all_label[index] = int(a[i])
                    index += 1
            all_label = all_label.reshape(19, 19)

        index2 = 0
        save_f = label_loc_path + fileName
        with open(save_f, "w") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
            for i in range(19 ):
                for j in range(19 - 1):
                    label = str(int(all_label[i][j])) + str(int(all_label[i][j + 1]))
                    label_bnd = label + ' ' + str(int(bndbox[index2][0])) + ' ' + str(int(bndbox[index2][1])) + ' ' + str(int(bndbox[index2][2])) + ' ' + str(int(bndbox[index2][3])) + '\n'
                    index2 += 1
                    file.write(label_bnd)

def generate_bndbox_txt_corner(corner4txt_path, label_loc_path):
    fileList = os.listdir(corner4txt_path)
    os.chdir(corner4txt_path)
    for fileName in fileList:  # 遍历文件夹中所有文件
        f_read = corner4txt_path + fileName

        print(f_read)
        with open(f_read, "r") as f:
            a = f.read().split('\t')  # 读取文件
            point4 = np.zeros(8)
            for i in range(8):
                point4[i] = int(a[i])
            point4 = np.reshape(point4, (4, 2))
            point4 = add_sort(point4)  # add sort
            boardgrid = getBoardGrid(point4, 19)
            bndbox = generateBndBox_corner(boardgrid)
            # print(bndbox)
            save_f = label_loc_path + fileName
            with open(save_f, "w") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
                line1 = 'corner ' + str(int(bndbox[0][0])) + ' ' + str(int(bndbox[0][1])) + ' ' + str(
                    int(bndbox[0][2])) + ' ' + str(int(bndbox[0][3])) + '\n'
                line2 = 'corner ' + str(int(bndbox[1][0])) + ' ' + str(int(bndbox[1][1])) + ' ' + str(
                    int(bndbox[1][2])) + ' ' + str(int(bndbox[1][3])) + '\n'
                line3 = 'corner ' + str(int(bndbox[2][0])) + ' ' + str(int(bndbox[2][1])) + ' ' + str(
                    int(bndbox[2][2])) + ' ' + str(int(bndbox[2][3])) + '\n'
                line4 = 'corner ' + str(int(bndbox[3][0])) + ' ' + str(int(bndbox[3][1])) + ' ' + str(
                    int(bndbox[3][2])) + ' ' + str(int(bndbox[3][3])) + '\n'
                file.write(line1)
                file.write(line2)
                file.write(line3)
                file.write(line4)

def generate_bndbox_txt_corner4(corner4txt_filepath, save_path):
    fileList = os.listdir(corner4txt_filepath)
    os.chdir(corner4txt_filepath)
    for fileName in fileList:  # 遍历文件夹中所有文件
        f_read = corner4txt_filepath + fileName
        print(f_read)
        with open(f_read, "r") as f:
            a = f.read().split('\t')  # 读取文件
            point4 = np.zeros(8)
            for i in range(8):
                point4[i] = int(a[i])
            point4 = np.reshape(point4, (4, 2))
            point4 = add_sort(point4)  # add sort
            boardgrid = getBoardGrid(point4, 19)
            bndbox = generateBndBox_corner(boardgrid)
            # print(bndbox)
            save_f = save_path + fileName
            with open(save_f, "w") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
                line1 = 'lefttop ' + str(int(bndbox[0][0])) + ' ' + str(int(bndbox[0][1])) + ' ' + str(
                    int(bndbox[0][2])) + ' ' + str(int(bndbox[0][3])) + '\n'
                line2 = 'righttop ' + str(int(bndbox[1][0])) + ' ' + str(int(bndbox[1][1])) + ' ' + str(
                    int(bndbox[1][2])) + ' ' + str(int(bndbox[1][3])) + '\n'
                line3 = 'leftbottom ' + str(int(bndbox[2][0])) + ' ' + str(int(bndbox[2][1])) + ' ' + str(
                    int(bndbox[2][2])) + ' ' + str(int(bndbox[2][3])) + '\n'
                line4 = 'rightbottom ' + str(int(bndbox[3][0])) + ' ' + str(int(bndbox[3][1])) + ' ' + str(
                    int(bndbox[3][2])) + ' ' + str(int(bndbox[3][3])) + '\n'
                file.write(line1)
                file.write(line2)
                file.write(line3)
                file.write(line4)

def generate_bndbox_txt_2_1(corner4txt_path, label_filepath, label_loc_path):
    fileList = os.listdir(corner4txt_path)
    os.chdir(corner4txt_path)
    for fileName in fileList:
        ## 获取 19 * 18 个 bndbox
        f_read = corner4txt_path + fileName
        print(f_read)
        with open(f_read, "r") as f:
            a = f.read().split('\t')  # 读取文件
        point4 = np.zeros(8)
        for i in range(8):
            point4[i] = int(a[i])
        point4 = np.reshape(point4, (4, 2))
        point4 = add_sort(point4)  # add sort
        boardgrid = getBoardGrid(point4, 19)
        bndbox_y, bndbox_x = getVOCBox(boardgrid)
        bndbox = generateBndBox_2_1(bndbox_y, bndbox_x)

        ## 获取 label
        label_filepath_f = label_filepath + fileName
        all_label = np.zeros(19 * 19) - 1
        index = 0
        with open(label_filepath_f) as f:
            a = f.read()
            for i in range(len(a)):
                if a[i] == '0' or a[i] == '1' or a[i] == '2':
                    all_label[index] = int(a[i])
                    index += 1
            all_label = all_label.reshape(19, 19)

        index2 = 0
        save_f = label_loc_path + fileName
        with open(save_f, "w") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
            for i in range(19 - 1):
                for j in range(19):
                    label = str(int(all_label[i][j])) + str(int(all_label[i+1][j]))
                    label_bnd = label + ' ' + str(int(bndbox[index2][0])) + ' ' + str(int(bndbox[index2][1])) + ' ' + str(int(bndbox[index2][2])) + ' ' + str(int(bndbox[index2][3])) + '\n'
                    index2 += 1
                    file.write(label_bnd)

# 输入的四个点格式为：lefttop, righttop, leftbottom, rightbottom
def main():
    corner4txt_path = r"F:/ZYY/chess/images/train/txt4corner/"
    label_path = "F:/ZYY/chess/images/train/label/"
    label_loc_path = "F:/ZYY/chess/images/train/Annotation/label_loc/"
    label_loc_path1 = "F:/ZYY/chess/images/train/Annotation/lab_loc/1-1/"
    label_loc_path2 = "F:/ZYY/chess/images/train/Annotation/lab_loc/1-2/"
    label_loc_path3 = "F:/ZYY/chess/images/train/Annotation/lab_loc/corner/"
    label_loc_path4 = "F:/ZYY/chess/images/train/Annotation/lab_loc/2-1/"

    # generate_bndbox_txt_1_1(corner4txt_path, label_path, label_loc_path1)
    # generate_bndbox_txt_1_2(corner4txt_path, label_path, label_loc_path2)
    # generate_bndbox_txt_corner(corner4txt_path, label_loc_path3)
    # generate_bndbox_txt_corner4(corner4txt_path, label_path, label_loc_path)
    # generate_bndbox_txt_2_1(corner4txt_path, label_path, label_loc_path4)

    corner4txt_path = r'F:/ZYY/chess/images/test/txt4corner_cut/'
    label_loc_path = "F:/ZYY/chess/images/test/anno/lab_loc/"
    generate_bndbox_txt_corner(corner4txt_path, label_loc_path)

if __name__ == '__main__':
    main()