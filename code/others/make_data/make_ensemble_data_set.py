# 由matirx值制作 dataset
import os, sys
import numpy as np


def data(matrix_11_path, matrix_12_path,real_matrix_path, save_path):
    # 1*1预测的marix , 1*2预测的marix, 真实的matrix

    fileList = os.listdir(real_matrix_path)
    os.chdir(real_matrix_path)
    file_num = 0
    err = 0
    for fileName in fileList:
        test_file = matrix_11_path + fileName
        test_file2 = matrix_12_path + fileName
        real_file = real_matrix_path + fileName

        ## 获取 19*19 label
        test_label = np.zeros(19 * 19) - 1
        index = 0
        with open(test_file) as f:
            a = f.read()
            for i in range(len(a)):
                # print(i)
                if a[i] == '0' or a[i] == '2':
                    test_label[index] = int(a[i])
                    index += 1
                elif a[i] == '1':
                    if a[i - 1] == '-':
                        test_label[index] = -1
                    else:
                        test_label[index] = 1
                    index += 1
            test_label = test_label.reshape(19, 19)
            f.close()

        test_label2 = np.zeros(19 * 19) - 1
        index = 0
        with open(test_file2) as f:
            a = f.read()
            for i in range(len(a)):
                # print(i)
                if a[i] == '0' or a[i] == '2':
                    test_label2[index] = int(a[i])
                    index += 1
                elif a[i] == '1':
                    if a[i - 1] == '-':
                        test_label2[index] = -1
                    else:
                        test_label2[index] = 1
                    index += 1
            test_label2 = test_label2.reshape(19, 19)
            f.close()

        real_label = np.zeros(19 * 19) - 1
        index = 0
        with open(real_file) as f:
            a = f.read()
            for i in range(len(a)):
                if a[i] == '0' or a[i] == '1' or a[i] == '2':
                    real_label[index] = int(a[i])
                    index += 1
            real_label = real_label.reshape(19, 19)
            f.close()

        with open(save_path, "a") as file:
            for i in range(19):
                for j in range(19):
                    line = str(int(test_label[i][j]))+'\t' + str(int(test_label2[i][j]))+'\t' + str(int(real_label[i][j])) + '\n'
                    file.write(line)
    file.close()

def data_4(matrix_11_path, matrix_12_path,real_matrix_path,conf_11_path,conf_12_path,save_path):
    fileList = os.listdir(matrix_11_path)
    os.chdir(matrix_11_path)

    file_num = 0
    err = 0
    for fileName in fileList:
        print(fileName)
        matrix_file1 = matrix_11_path + fileName
        matrix_file2 = matrix_12_path + fileName
        real_file = real_matrix_path + fileName
        matrix_conf1 = conf_11_path + fileName
        matrix_conf2 = conf_12_path + fileName

        ## 获取 label
        label1 = np.zeros(19 * 19) - 1
        index = 0
        with open(matrix_file1) as f:
            a = f.read()
            for i in range(len(a)):
                # print(i)
                if a[i] == '0' or a[i] == '2':
                    label1[index] = int(a[i])
                    index += 1
                elif a[i] == '1':
                    if a[i - 1] == '-':
                        label1[index] = -1
                    else:
                        label1[index] = 1
                    index += 1
            label1 = label1.reshape(19, 19)
            f.close()

        label2 = np.zeros(19 * 19) - 1
        index = 0
        with open(matrix_file2) as f:
            a = f.read()
            for i in range(len(a)):
                # print(i)
                if a[i] == '0' or a[i] == '2':
                    label2[index] = int(a[i])
                    index += 1
                elif a[i] == '1':
                    if a[i - 1] == '-':
                        label2[index] = -1
                    else:
                        label2[index] = 1
                    index += 1
            label2 = label2.reshape(19, 19)
            f.close()

        real_label = np.zeros(19 * 19) - 1
        index = 0
        with open(real_file) as f:
            a = f.read()
            for i in range(len(a)):
                if a[i] == '0' or a[i] == '1' or a[i] == '2':
                    real_label[index] = int(a[i])
                    index += 1
            real_label = real_label.reshape(19, 19)
            f.close()

        conf_11 = np.zeros(19 * 19) - 1
        index = 0
        with open(matrix_conf1) as f:
            a = f.read()
            for i in range(len(a)):
                if a[i] == '.':
                    conf_11[index] = float(a[i - 1] + a[i] + a[i + 1] + a[i + 2])
                    index += 1
            conf_11 = conf_11.reshape(19, 19)
            f.close()

        conf_12 = np.zeros(19 * 19) - 1
        index = 0
        with open(matrix_conf2) as f:
            a = f.read()
            for i in range(len(a)):
                if a[i] == '.':
                    conf_12[index] = float(a[i - 1] + a[i] + a[i + 1] + a[i + 2])
                    index += 1
            conf_12 = conf_12.reshape(19, 19)
            f.close()

        with open(save_path, "a") as file:
            for i in range(19):
                for j in range(19):
                    line = str(int(label1[i][j]))+'\t' + \
                           str(float(conf_11[i][j]))+'\t' +\
                           str(int(label2[i][j]))+'\t' + \
                           str(float(conf_12[i][j])) + '\t' + \
                           str(int(real_label[i][j])) + '\n'
                    file.write(line)
    file.close()

def data_6(matrix_11_path, matrix_12_path, matrix_21_path, real_matrix_path, conf_11_path, conf_12_path, conf_21_path, save_path):
    fileList = os.listdir(matrix_11_path)
    os.chdir(matrix_11_path)

    file_num = 0
    err = 0
    for fileName in fileList:
        print(fileName)
        matrix_file1 = matrix_11_path + fileName
        matrix_file2 = matrix_12_path + fileName
        matrix_file3 = matrix_21_path + fileName
        real_file = real_matrix_path + fileName
        matrix_conf1 = conf_11_path + fileName
        matrix_conf2 = conf_12_path + fileName
        matrix_conf3 = conf_21_path + fileName

        ## 获取 label
        label1 = np.zeros(19 * 19) - 1
        index = 0
        with open(matrix_file1) as f:
            a = f.read()
            for i in range(len(a)):
                # print(i)
                if a[i] == '0' or a[i] == '2':
                    label1[index] = int(a[i])
                    index += 1
                elif a[i] == '1':
                    if a[i - 1] == '-':
                        label1[index] = -1
                    else:
                        label1[index] = 1
                    index += 1
            label1 = label1.reshape(19, 19)
            f.close()

        label2 = np.zeros(19 * 19) - 1
        index = 0
        with open(matrix_file2) as f:
            a = f.read()
            for i in range(len(a)):
                # print(i)
                if a[i] == '0' or a[i] == '2':
                    label2[index] = int(a[i])
                    index += 1
                elif a[i] == '1':
                    if a[i - 1] == '-':
                        label2[index] = -1
                    else:
                        label2[index] = 1
                    index += 1
            label2 = label2.reshape(19, 19)
            f.close()

        label3 = np.zeros(19 * 19) - 1
        index = 0
        with open(matrix_file3) as f:
            a = f.read()
            for i in range(len(a)):
                # print(i)
                if a[i] == '0' or a[i] == '2':
                    label3[index] = int(a[i])
                    index += 1
                elif a[i] == '1':
                    if a[i - 1] == '-':
                        label3[index] = -1
                    else:
                        label3[index] = 1
                    index += 1
            label3 = label3.reshape(19, 19)
            f.close()

        real_label = np.zeros(19 * 19) - 1
        index = 0
        with open(real_file) as f:
            a = f.read()
            for i in range(len(a)):
                if a[i] == '0' or a[i] == '1' or a[i] == '2':
                    real_label[index] = int(a[i])
                    index += 1
            real_label = real_label.reshape(19, 19)
            f.close()

        conf_11 = np.zeros(19 * 19) - 1
        index = 0
        with open(matrix_conf1) as f:
            a = f.read()
            for i in range(len(a)):
                if a[i] == '.':
                    conf_11[index] = float(a[i - 1] + a[i] + a[i + 1] + a[i + 2])
                    index += 1
            conf_11 = conf_11.reshape(19, 19)
            f.close()

        conf_12 = np.zeros(19 * 19) - 1
        index = 0
        with open(matrix_conf2) as f:
            a = f.read()
            for i in range(len(a)):
                if a[i] == '.':
                    conf_12[index] = float(a[i - 1] + a[i] + a[i + 1] + a[i + 2])
                    index += 1
            conf_12 = conf_12.reshape(19, 19)
            f.close()

        conf_21 = np.zeros(19 * 19) - 1
        index = 0
        with open(matrix_conf3) as f:
            a = f.read()
            for i in range(len(a)):
                if a[i] == '.':
                    conf_21[index] = float(a[i - 1] + a[i] + a[i + 1] + a[i + 2])
                    index += 1
            conf_21 = conf_21.reshape(19, 19)
            f.close()

        with open(save_path, "a") as file:
            for i in range(19):
                for j in range(19):
                    line = str(int(label1[i][j])) + '\t' + \
                           str(float(conf_11[i][j])) + '\t' + \
                           str(int(label2[i][j])) + '\t' + \
                           str(float(conf_12[i][j])) + '\t' + \
                           str(int(label3[i][j])) + '\t' + \
                           str(float(conf_21[i][j])) + '\t' + \
                           str(int(real_label[i][j])) + '\n'
                    file.write(line)
    file.close()


def main():
    real_matrix_path = 'F:/ZYY/chess/images/train/label/'
    matrix_11_path_train = "F:/ZYY/chess/model_ensemble/data/train/matrix-1-1/"
    matrix_12_path_train = "F:/ZYY/chess/model_ensemble/data/train/matrix-1-2/"
    matrix_21_path_train = "F:/ZYY/chess/model_ensemble/data/train/matrix-2-1/"
    conf_11_path_train = "F:/ZYY/chess/model_ensemble/data/train/conf-1-1/"
    conf_12_path_train = "F:/ZYY/chess/model_ensemble/data/train/conf-1-2/"
    conf_21_path_train = "F:/ZYY/chess/model_ensemble/data/train/conf-2-1/"
    save_path_train_11_12 = "F:/ZYY/chess/model_ensemble/data/train/train_data11_12.txt"
    save_path_train_11_21 = "F:/ZYY/chess/model_ensemble/data/train/train_data11_21.txt"
    save_path_6_train = "F:/ZYY/chess/model_ensemble/data/train/train_data6.txt"

    matrix_11_path_test = "F:/ZYY/chess/model_ensemble/data/test/matrix-1-1/"
    matrix_12_path_test = "F:/ZYY/chess/model_ensemble/data/test/matrix-1-2/"
    matrix_21_path_test = "F:/ZYY/chess/model_ensemble/data/test/matrix-1-2/"
    conf_11_path_test = "F:/ZYY/chess/model_ensemble/data/test/conf-1-1/"
    conf_12_path_test = "F:/ZYY/chess/model_ensemble/data/test/conf-1-2/"
    conf_21_path_test = "F:/ZYY/chess/model_ensemble/data/test/conf-2-1/"
    save_path_test_11_12 = "F:/ZYY/chess/model_ensemble/data/train/test_data11_12.txt"
    save_path_test_11_21 = "F:/ZYY/chess/model_ensemble/data/train/test_data11_21.txt"
    save_path_6_test = "F:/ZYY/chess/model_ensemble/data/train/test_data6.txt"

    data_4(matrix_11_path_train, matrix_12_path_train, real_matrix_path, conf_11_path_train,conf_12_path_train,save_path_train_11_12)
    data_4(matrix_11_path_test, matrix_12_path_test, real_matrix_path, conf_11_path_test, conf_12_path_test,save_path_test_11_12)
    data_4(matrix_11_path_train, matrix_21_path_train, real_matrix_path, conf_11_path_train, conf_21_path_train,save_path_train_11_21)
    data_4(matrix_11_path_test, matrix_21_path_test, real_matrix_path, conf_11_path_test, conf_21_path_test,save_path_test_11_21)

    # data_4(matrix_12_path_train, matrix_21_path_train, real_matrix_path, conf_12_path_train, conf_21_path_train,save_path_train_12_21)
    # data_4(matrix_12_path_test, matrix_21_path_test, real_matrix_path, conf_21_path_test, conf_12_path_test,save_path_test_12_21)

    # data_6(matrix_11_path_train, matrix_12_path_train, matrix_21_path_train, real_matrix_path, conf_11_path_train, conf_12_path_train, conf_21_path_train, save_path_6_train)
    # data_6(matrix_11_path_test, matrix_12_path_test, matrix_21_path_test, real_matrix_path, conf_11_path_test,conf_12_path_test, conf_21_path_test, save_path_6_test)


if __name__ == '__main__':
    main()
    a = 1