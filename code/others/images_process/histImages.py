import cv2 as cv
import numpy as np
import os
# https://blog.csdn.net/zjy123078_zjy/article/details/105730171




def EqualHist(src):
    # src = cv.cvtColor(src, cv.COLOR_BGR2GRAY) # 灰度化
    # src = np.sqrt(src) * 16
    dst = cv.equalizeHist(src)  # 全局直方图均衡化 # dst: 输出图像矩阵(src的shape一样)
    # clahe = cv.createCLAHE(clipLimit=5, tileGridSize=(7, 7)) # 局部直方图均衡化
    # dst = clahe.apply(gray)
    # cv.imshow("global equalizeHist", dst)
    return dst


def main():
    images_path = "F:/ZYY/chess/images/test3/check/labs/images/"
    save_path = "F:/ZYY/chess/images/test3/check/labs/hist/"

    fileList = os.listdir(images_path)
    for fileName in fileList:
        print(fileName)
        path = images_path + fileName
        src = cv.imread(path)

        src = cv.cvtColor(src, cv.COLOR_BGR2GRAY) # 灰度化
        # src = np.sqrt(src) * 16
        dst = cv.equalizeHist(src)  # 全局直方图均衡化 # dst: 输出图像矩阵(src的shape一样)
        # clahe = cv.createCLAHE(clipLimit=5, tileGridSize=(7, 7)) # 局部直方图均衡化
        # dst = clahe.apply(gray)
        # cv.imshow("global equalizeHist", dst)
        cv.imwrite(save_path + fileName, dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()