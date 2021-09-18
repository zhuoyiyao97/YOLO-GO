# 检测matrix与real_matrix之间的比较
import os, sys
import numpy as np

def compare(real_matrix_path, test_matrix_path, txt_path):
    fileList = os.listdir(test_matrix_path)
    os.chdir(test_matrix_path)

    file_num = 0
    err = 0
    for fileName in fileList:
        # with open(txt_path, "a") as file:
        #     file.write(fileName + '\n')

        print(fileName)
        test_file = test_matrix_path + fileName
        real_file = real_matrix_path + fileName
        file_num += 1

        ## 获取 label
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

        for i in range(19):
            for j in range(19):
                if int(test_label[i][j]) != int(real_label[i][j]):
                    err += 1
                    line = "位置：(" + str(i+1)+',' + str(j+1)+ ')'+ ";    检测值= "+ str(test_label[i][j])+ ";    真实值= "+str(real_label[i][j])
                    # with open(txt_path, "a") as file:
                    #     file.write(line)
                    print(line)


    print("err = ", str(err), "\tall = ", str(361 * file_num), "\tacc = ", str((361 * file_num - err) / 361 / file_num))
    # with open(txt_path, "a") as file:
        # file.write(str(err) + "acc = " + str((361 * file_num - err) / 361 / file_num))
    # file.close()

def main():
    real_matrix_path = "F:/ZYY/chess/images/test2/labels/"
    test_matrix_path = "F:/ZYY/chess/images/test2/tmp/matrix-ensemble/"
    txt_path = 'F:/ZYY/chess/model_ensemble/data/test/TXT/tmp.txt'
    compare(real_matrix_path, test_matrix_path, txt_path)

    real_matrix_path = "F:/ZYY/chess/论文/result/label/"
    test_matrix_path = "F:/ZYY/chess/论文/result/matrix-1-1/"
    txt_path = 'F:/ZYY/chess/model_ensemble/data/test/TXT/tmp.txt'
    compare(real_matrix_path, test_matrix_path, txt_path)

if __name__ == '__main__':
    main()