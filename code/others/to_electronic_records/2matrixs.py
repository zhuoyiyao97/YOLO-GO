# 把label loc根据四角坐标映射成matrix的形式
# 待改进
# 没有检测到的填-1

import os, sys
import numpy as np
import cv2
WY_BOARD_LINES = 19

# boardgrid 361 * 2
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

# data bndNum * 5 (label,x, y, conf, flag)
def read_out_txt(f_read):
    list_a = open(f_read).read().splitlines()
    line_num = len(list_a)
    print(line_num)
    # print("漏检数： " + str(361-line_num))
    data = np.zeros(line_num * 5)
    i = 0
    for line in list_a:
        line = line.split(' ') # label,xmin, ymin, xmax, ymax
        data[i] = int(line[0])
        data[i + 1] = int((int(line[1]) + int(line[3]) ) / 2)
        data[i + 2] = int((int(line[2]) + int(line[4]) ) / 2)
        data[i + 3] = float(line[5])
        data[i + 4] = 1
        i = i + 5
    data = np.reshape(data, (-1, 5))
    return data

def corner_sort(point4):
    dict = {point4[0][0]: point4[0][1], point4[1][0]: point4[1][1], point4[2][0]: point4[2][1],
            point4[3][0]: point4[3][1]}  # (x,y)
    my_list = [point4[0][0], point4[1][0], point4[2][0], point4[3][0]]
    my_list.sort()
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

def order_1_1(boardgrid, data):
    labels = np.zeros([19, 19]) - 1
    conf = np.zeros([19, 19]) - 1
    for i in range(19):
        for j in range(19):
            min_dis = 100000000
            min_index = -1
            x_truth = boardgrid[i * 19 + j][0]
            y_truth = boardgrid[i * 19 + j][1]
            for index in range(data.shape[0]): # 返回欧氏距离最小的下标
                # if data[index][3] == 0:
                #     continue
                if data[index][0] == 3:
                    continue
                else:
                    x_dect = data[index][1]
                    y_dect = data[index][2]
                    distance = pow(x_truth - x_dect, 2) + pow(y_truth-y_dect,2)
                    if min_dis > distance:
                        min_dis = distance
                        min_index = index
            # labels[i][j] = data[min_index][0]
            if j == 18:
                x_near = boardgrid[i*19 + j - 1][0]
                y_near = boardgrid[i * 19 + j - 1][1]
            else:
                x_near = boardgrid[i * 19 + j +1][0]
                y_near = boardgrid[i * 19 + j + 1][1]
            near_dis = pow(x_truth - x_near, 2) + pow(y_truth-y_near,2)
            if min_dis > near_dis/8: # 说明该格子没有检测到
                labels[i][j] = -1
                conf[i][j] = 0
            else: # 网络练叉了...
                # if data[min_index][0] == 0:
                #     labels[i][j] = 2
                # elif data[min_index][0] == 2:
                #     labels[i][j] = 0
                # elif data[min_index][0] == 1:
                #     labels[i][j] = 1
                labels[i][j] = data[min_index][0]
                conf[i][j] = data[min_index][3]
            # data[min_index][3] = 0
            # print(19*i + j,data[min_index])
    return labels, conf

def order_1_2(boardgrid, data):
    labels = np.zeros([19, 19]) - 1
    conf = np.zeros([19, 19]) - 1
    for i in range(19):
        for j in range(19):
            min_dis = [100000000,100000000]
            min_index = [-1, -1]
            x_truth = boardgrid[i * 19 + j][0]
            y_truth = boardgrid[i * 19 + j][1]
            for index in range(data.shape[0]): # 返回欧氏距离最小的2个下标
                if data[index][3] == 0:
                    continue
                elif data[index][0] == 3:
                    continue
                else:
                    x_dect = data[index][1]
                    y_dect = data[index][2]
                    distance = pow(x_truth - x_dect, 2) + pow(y_truth-y_dect,2)
                    if distance < min_dis[0]: # 最小
                        min_dis[0] = distance
                        min_index[0] = index
                    elif distance < min_dis[1]: # 次小
                        min_dis[1] = distance
                        min_index[1] = index

            if j == 18:
                x_near = boardgrid[i*19 + j - 1][0]
                y_near = boardgrid[i * 19 + j - 1][1]
            else:
                x_near = boardgrid[i * 19 + j +1][0]
                y_near = boardgrid[i * 19 + j + 1][1]
            near_dis = pow(x_truth - x_near, 2) + pow(y_truth-y_near,2) # 所在的两个格子之间的距离
            if min_dis[0] < near_dis/8 and min_dis[1] < near_dis/16: # 两个,以置信度高的为准
                if data[min_index[0]][3] > data[min_index[1]][3]:
                    labels[i][j] = data[min_index[0]][0]
                    conf[i][j] = data[min_index[0]][3]
                else:
                    labels[i][j] = data[min_index[1]][0]
                    conf[i][j] = data[min_index[1]][3]
            elif min_dis[0] < near_dis/16 and min_dis[1] > near_dis/16: # 一个
                labels[i][j] = data[min_index[0]][0]
                conf[i][j] = data[min_index[0]][3]
            else:
                labels[i][j] = -1
                conf[i][j] = 0
            # data[min_index][3] = 0
            # print(19*i + j,data[min_index])
    return labels, conf

def matrix_1_1_label(label_loc_path, corner4txt_path, save_path):
    fileList = os.listdir(label_loc_path)
    # os.chdir(label_loc_path)
    for fileName in fileList:
        save_file = save_path + fileName
        f_read = corner4txt_path + fileName
        print(f_read)
        with open(f_read, "r") as f:
            a = f.read().split('\t')  # 读取文件
        point4 = np.zeros(8)
        for i in range(8):
            point4[i] = int(a[i])
        point4 = np.reshape(point4, (4, 2))
        point4 = corner_sort(point4)
        boardgrid = getBoardGrid(point4, 19)

        f_read = label_loc_path + fileName
        data = read_out_txt(f_read)
        labels,conf = order_1_1(boardgrid, data)

        with open(save_file, "w") as file:
            file.write(str(labels))
            file.close()


def matrix_1_2_label(label_loc_path, corner4txt_path, save_path):
    fileList = os.listdir(label_loc_path)
    # os.chdir(label_loc_path)
    for fileName in fileList:
        save_file = save_path + fileName
        f_read = corner4txt_path + fileName
        print(f_read)
        with open(f_read, "r") as f:
            a = f.read().split('\t')  # 读取文件
        point4 = np.zeros(8)
        for i in range(8):
            point4[i] = int(a[i])
        point4 = np.reshape(point4, (4, 2))
        point4 = corner_sort(point4)
        boardgrid = getBoardGrid(point4, 19)

        f_read = label_loc_path + fileName
        data = read_out_txt(f_read)
        labels, conf = order_1_2(boardgrid, data)

        with open(save_file, "w") as file2:
            file2.write(str(labels))
        file2.close()


def matrix_1_1_conf(label_loc_path, corner4txt_path, save_path):
    fileList = os.listdir(label_loc_path)
    # os.chdir(label_loc_path)
    for fileName in fileList:
        save_file = save_path + fileName
        f_read = corner4txt_path + fileName
        print(f_read)
        with open(f_read, "r") as f:
            a = f.read().split('\t')  # 读取文件
        point4 = np.zeros(8)
        for i in range(8):
            point4[i] = int(a[i])
        point4 = np.reshape(point4, (4, 2))
        point4 = corner_sort(point4)
        boardgrid = getBoardGrid(point4, 19)

        f_read = label_loc_path + fileName
        data = read_out_txt(f_read)
        labels,conf = order_1_1(boardgrid, data)

        with open(save_file, "w") as file:
            file.write(str(conf))
            file.close()

def matrix_1_2_conf(label_loc_path, corner4txt_path, save_path):
    fileList = os.listdir(label_loc_path)
    # os.chdir(label_loc_path)
    for fileName in fileList:
        save_file = save_path + fileName
        f_read = corner4txt_path + fileName
        print(f_read)
        with open(f_read, "r") as f:
            a = f.read().split('\t')  # 读取文件
        point4 = np.zeros(8)
        for i in range(8):
            point4[i] = int(a[i])
        point4 = np.reshape(point4, (4, 2))
        point4 = corner_sort(point4)
        boardgrid = getBoardGrid(point4, 19)

        f_read = label_loc_path + fileName
        data = read_out_txt(f_read)
        labels, conf = order_1_2(boardgrid, data)

        # 制作conf
        with open(save_file, "w") as file:
            file.write(str(conf))
        file.close()

def main():
    # label_loc_path = "F:/ZYY/chess/model_ensemble/result/test-1-1-txt/"
    # label_loc_path2 = "F:/ZYY/chess/model_ensemble/result/test-1-2-txt/"
    # corner4txt_path = r"F:/ZYY/chess/model_ensemble/data/txt4corner/"
    # save_path = "F:/ZYY/chess/model_ensemble/result/matrix-1-1/"
    # save_path2 = "F:/ZYY/chess/model_ensemble/result/matrix-1-2/"

    # if os.path.exists(save_path1) == False:  # 不存在則創建文件夾
    #     os.makedirs(save_path1)
    # if os.path.exists(save_path2) == False:  # 不存在則創建文件夾
    #     os.makedirs(save_path2)

    corner4txt_path = r"F:/ZYY/chess/images/train/txt4corner/"
    corner4txt_path = "F:/ZYY/chess/images/train/NOCUTS_0413/txt4corner/"


    train_path = "F:/ZYY/chess/model_ensemble/data/train/"
    test_path = "F:/ZYY/chess/model_ensemble/data/test/"

    label_loc_11_train = train_path + "train-1-1-txt/"
    label_loc_12_train = train_path + "train-1-2-txt/"
    label_loc_21_train = train_path + "train-2-1-txt/"
    save_path_11_train = train_path + "matrix-1-1/"
    save_path_12_train = train_path + "matrix-1-2/"
    save_path_21_train = train_path + "matrix-2-1/"
    save_conf_11_train = train_path + "conf-1-1/"
    save_conf_12_train = train_path + "conf-1-2/"
    save_conf_21_train = train_path + "conf-2-1/"

    label_loc_11_test = test_path + "test-1-1-txt/"
    label_loc_12_test = test_path + "test-1-2-txt/"
    label_loc_21_test = test_path + "test-2-1-txt/"
    save_path_11_test = test_path + "matrix-1-1/"
    save_path_12_test = test_path + "matrix-1-2/"
    save_path_21_test = test_path + "matrix-2-1/"
    save_conf_11_test = test_path + "conf-1-1/"
    save_conf_12_test = test_path + "conf-1-2/"
    save_conf_21_test = test_path + "conf-2-1/"

    # matrix_1_1_label(label_loc_11_train, corner4txt_path, save_path_11_train)
    # matrix_1_1_label(label_loc_11_test, corner4txt_path, save_path_11_test)
    # matrix_1_1_conf(label_loc_11_train, corner4txt_path, save_conf_11_train)
    # matrix_1_1_conf(label_loc_11_test, corner4txt_path, save_conf_11_test)

    # matrix_1_2_label(label_loc_12_train, corner4txt_path, save_path_12_train)
    # matrix_1_2_label(label_loc_12_test, corner4txt_path, save_path_12_test)
    # matrix_1_2_conf(label_loc_12_train, corner4txt_path, save_conf_12_train)
    # matrix_1_2_conf(label_loc_12_test, corner4txt_path, save_conf_12_test)

    # matrix_1_2_label(label_loc_21_train, corner4txt_path, save_path_21_train)
    # matrix_1_2_label(label_loc_21_test, corner4txt_path, save_path_21_test)
    # matrix_1_2_conf(label_loc_21_train, corner4txt_path, save_conf_21_train)
    # matrix_1_2_conf(label_loc_21_test, corner4txt_path, save_conf_21_test)

    # 0511
    # save_path_11 = "F:/ZYY/chess/images/test2/check/matrix-1-1/"
    # label_loc_11 = "F:/ZYY/chess/images/test2/check/txt-1-1/"
    # corner4txt_path = 'F:/ZYY/chess/images/test2/txt4corner/'
    # matrix_1_1_label(label_loc_11, corner4txt_path, save_path_11)
    #
    # save_path_12 = "F:/ZYY/chess/images/test2/check/matrix-1-2/"
    # label_loc_12 = "F:/ZYY/chess/images/test2/check/txt-1-2/"
    # corner4txt_path = 'F:/ZYY/chess/images/test2/txt4corner/'
    # matrix_1_2_label(label_loc_12, corner4txt_path, save_path_12)


    save_path_11 = "F:/ZYY/chess/images/test3/check/labs/matrix-1-1/"
    label_loc_11 = "F:/ZYY/chess/images/test3/check/labs/txt-1-1/"
    corner4txt_path = 'F:/ZYY/chess/images/test3/txt4corner/txt4corner-trans/'
    matrix_1_1_label(label_loc_11, corner4txt_path, save_path_11)

    save_path_12 = "F:/ZYY/chess/images/test3/check/labs/matrix-1-2/"
    label_loc_12 = "F:/ZYY/chess/images/test3/check/labs/txt-1-2/"
    matrix_1_2_label(label_loc_12, corner4txt_path, save_path_12)

if __name__ == '__main__':
    main()
