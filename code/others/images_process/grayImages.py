from PIL import Image
import os
import glob
from numpy import *
import cv2 as cv
import numpy as np

def histeq(im,nbr_bins = 256):
    """对一幅灰度图像进行直方图均衡化"""
    #计算图像的直方图
    #在numpy中，也提供了一个计算直方图的函数histogram(),第一个返回的是直方图的统计量，第二个为每个bins的中间值
    im = array(im,'f')
    imhist,bins = histogram(im.flatten(),nbr_bins,normed= True)
    cdf = imhist.cumsum()   #
    cdf = 255.0 * cdf / cdf[-1]
    #使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf


def localEqualHist(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=5, tileGridSize=(7,7))
    dst = clahe.apply(gray)
    # cv.imshow("clahe image", dst)

def gray_his_images2(input_dir, out_dir1, out_dir2):
    img_Lists = glob.glob(input_dir + '/*.jpg')
    img_basenames = []  # e.g. 100.jpg
    for item in img_Lists:
        img_basenames.append(os.path.basename(item))

    img_names = []  # e.g. 100
    for item in img_basenames:
        temp1, temp2 = os.path.splitext(item)
        img_names.append(temp1)
    a = os.listdir(input_dir)

    for i in a:
        print(i)
        i = input_dir + i
        img = Image.open(i)
        img = 255 * np.array(img).astype('uint8')
        gray = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)  # PIL转cv2
        # 全局直方图均衡化
        dst1 = cv.equalizeHist(gray)  # dst: 输出图像矩阵(src的shape一样)
        dst1.save(out_dir1 + i)
        # 局部
        clahe = cv.createCLAHE(clipLimit=5, tileGridSize=(7, 7))
        dst2 = clahe.apply(gray)
        dst2.save(out_dir2 + i)

def gray_his_images(input_dir, out_dir):
    img_Lists = glob.glob(input_dir + '/*.jpg')
    img_basenames = []  # e.g. 100.jpg
    for item in img_Lists:
        img_basenames.append(os.path.basename(item))
    
    img_names = []  # e.g. 100
    for item in img_basenames:
        temp1, temp2 = os.path.splitext(item)
        img_names.append(temp1)
    a = os.listdir(input_dir)
    
    for i in a:
        print(i)
        src = Image.open(input_dir + i)
        # 灰度化
        src = src.convert('L')
        # L.save(out_dir1 + i)
        # 非线性变换
        src = np.sqrt(src) * 16 # 法一
        # src = np.square(src) / 266
        # 直方图均衡化
        # src, cdf = histeq(src)
        src = Image.fromarray(uint8(src))
        src.save(out_dir + i)

if __name__ == '__main__':
    # 图像存储位置.*/
    input_dir = "F:/ZYY/chess/images/test3/check/labs/images/"
    # 文件存放位置.*/F:\ZYY\chess\
    out_dir = "F:/ZYY/chess/images/test3/check/labs/hist/"

    # 图像存储位置.*/
    input_dir = "F:/ZYY/chess/images/test3/check/labs/hist/"
    # 文件存放位置.*/F:\ZYY\chess\
    out_dir = "F:/ZYY/chess/images/test3/check/labs/hist2/"

    gray_his_images(input_dir, out_dir)

