# 由matirx值制作 dataset
import os, sys
import numpy as np

def get_laebl (matrix_file1, matrix_file2):
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

    # real_label = np.zeros(19 * 19) - 1
    # index = 0
    # with open(real_file) as f:
    #     a = f.read()
    #     for i in range(len(a)):
    #         if a[i] == '0' or a[i] == '1' or a[i] == '2':
    #             real_label[index] = int(a[i])
    #             index += 1
    #     real_label = real_label.reshape(19, 19)

    return label1,label2

def main():
    real_matrix_path = 'F:/ZYY/chess/images/test3/labels/'
    matrix_11_path = "F:/ZYY/chess/images/test3/check/labs/matrix-1-1/"
    matrix_12_path = "F:/ZYY/chess/images/test3/check/labs/matrix-1-2/"
    save_path = "F:/ZYY/chess/images/test3/check/labs/matrix-ensemble/"

    # matrix_11_path = "F:/ZYY/chess/images/test2/tmp/matrix-1-1/"
    # matrix_12_path = "F:/ZYY/chess/images/test2/tmp/matrix-1-2/"
    # save_path = "F:/ZYY/chess/images/test2/tmp/matrix-ensemble/"

    fileList = os.listdir(matrix_11_path)
    os.chdir(matrix_11_path)
    err = 0
    fileNum = 0
    result = np.zeros([19 ,19])
    for fileName in fileList:
        print(fileName)
        fileNum += 1
        matrix_file1 = matrix_11_path + fileName
        matrix_file2 = matrix_12_path + fileName

        # real_file = real_matrix_path + fileName
        save_file = save_path + fileName
        label1, label2 = get_laebl (matrix_file1, matrix_file2)
        # 直接判断
        for i in range(19):
            for j in range(19):
                if int(label1[i][j]) != -1:
                    result[i][j] = int(label1[i][j])
                elif int(label2[i][j]) != -1:
                    result[i][j] = int(label2[i][j])

        with open(save_file, "w") as file:

            file.write(str(result))
            file.close()

if __name__ == '__main__':
    main()