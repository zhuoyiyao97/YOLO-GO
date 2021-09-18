import os
import re
import sys

path = r"F:/ZYY/github/experimental results/YOLO_GO/GO_PIECEX2"

fileList = os.listdir(path)  # 待修改文件夹
print("修改前：" + str(fileList))  # 输出文件夹中包含的文件
os.chdir(path)  # 将当前工作目录修改为待修改文件夹的位置
num = 1 # 名称变量
for fileName in fileList:  # 遍历文件夹中所有文件
    pat = ".+\.(jpg|xml|txt|JPG|png)"  # 匹配文件名正则表达式
    pattern = re.findall(pat, fileName)  # 进行匹配
    # num = int(fileName.split('.')[0]) + 3186
    print('num：', num, 'filename:', fileName)
    if num < 10:
        os.rename(fileName, ('00' + str(num) + '.' + pattern[0]) ) # 文件重新命名
    elif num < 100:
        os.rename(fileName, ('0' + str(num) + '.' + pattern[0]))  # 文件重新命名
    # elif num < 1000:
    #     os.rename(fileName, ('000' + str(num) + '.' + pattern[0]))  # 文件重新命名
    else:
        os.rename(fileName, (str(num) + '.' + pattern[0]))  # 文件重新命名
    num += 1
print("---------------------------------------------------")
sys.stdin.flush()  # 刷新
print("修改后：" + str(os.listdir(path)))  # 输出修改后文件夹中包含的文件


# 把IPG 改成 jpg
# path = r"F:/ZYY/chess/images/test"
# fileList = os.listdir(path)  # 待修改文件夹
# print("修改前：" + str(fileList))  # 输出文件夹中包含的文件
# os.chdir(path)  # 将当前工作目录修改为待修改文件夹的位置
# # num = 151 # 名称变量
# for fileName in fileList:  # 遍历文件夹中所有文件
#     pat = ".+\.(jpg|xml|txt|JPG)"  # 匹配文件名正则表达式
#     pattern = re.findall(pat, fileName)  # 进行匹配
#     if pattern[0] == "JPG":
#         print(fileName)
#         print(pattern[0])
#         os.rename(fileName, (fileName.split('.')[0]+ '.jpg'))  # 文件重新命名
# print("---------------------------------------------------")
# sys.stdin.flush()  # 刷新
# print("修改后：" + str(os.listdir(path)))  # 输出修改后文件夹中包含的文件

# path = r"F:/ZYY/chess/yolov5/yolov5-1-1/inference/images/new/"
# fileList = os.listdir(path)  # 待修改文件夹
# print("修改前：" + str(fileList))  # 输出文件夹中包含的文件
# os.chdir(path)  # 将当前工作目录修改为待修改文件夹的位置
# num = 3054 # 名称变量
# for fileName in fileList:  # 遍历文件夹中所有文件
#     pat = ".+\.(jpg|xml|txt|JPG|png)"  # 匹配文件名正则表达式
#     pattern = re.findall(pat, fileName)  # 进行匹配
#     # print('pattern[0]:', pattern)
#     print('num：', num, 'filename:', fileName)
#     if num < 10:
#         os.rename(fileName, ('00000' + str(num) + '.' + pattern[0]) ) # 文件重新命名
#     elif num < 100:
#         os.rename(fileName, ('0000' + str(num) + '.' + pattern[0]))  # 文件重新命名
#     elif num < 1000:
#         os.rename(fileName, ('000' + str(num) + '.' + pattern[0]))  # 文件重新命名
#     else:
#         os.rename(fileName, ('00' + str(num) + '.' + pattern[0]))  # 文件重新命名
#     # os.rename(fileName, ('000' + fileName) ) # 文件重新命名
#     # os.rename(fileName, ('name' + str(num) + '_' + '19970326' + '_中国海南' + '.' + pattern[0]))
#     num = num + 1  # 改变编号，继续下一项
# print("---------------------------------------------------")
# sys.stdin.flush()  # 刷新
# print("修改后：" + str(os.listdir(path)))  # 输出修改后文件夹中包含的文件