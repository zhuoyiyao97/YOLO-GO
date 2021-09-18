# 把文件2的内容接在文件1的后面
import os
import re
import sys


def concat(file1, file2):
    fileList = os.listdir(file2)  # 待修改文件夹
    # os.chdir(file2)  # 将当前工作目录修改为待修改文件夹的位置
    for fileName in fileList:  # 遍历文件夹中所有文件
        f1 = file1 + fileName
        f2 = file2 + fileName
        with open(f2, "r") as file:
            a = file.read()
        file.close()
        with open(f1, "a") as file:
            file.write(a)
        file.close()
        print(fileName)
    sys.stdin.flush()  # 刷新

# file1 = r"F:/ZYY/chess/images/images1368-3072/lab_loc/1-1-corner/"
# file2 = r"F:/ZYY/chess/images/images1368-3072/lab_loc/1-2-corner/"
# file = "F:/ZYY/chess/images/images1368-3072/lab_loc/corner/"

file = "F:/ZYY/chess/images/train/Annotation/lab_loc/corner/"
file1 = r"F:/ZYY/chess/images/train/Annotation/lab_loc/1-1-corner/"
file2 = r"F:/ZYY/chess/images/train/Annotation/lab_loc/1-2-corner/"
# file3 = r"F:/ZYY/chess/images/train/Annotation/lab_loc/2-1-corner/"

concat(file1, file) # 后面接在前
concat(file2, file)
# concat(file3, file) # 后面接在前

