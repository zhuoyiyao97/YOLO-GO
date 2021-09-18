import os
import cv2
import numpy as np

# image_file = "F:/ZYY/chess/images/train/NOCUTS_0413/images/gray/"
# image_save = "F:/ZYY/chess/images/train/images/gray/"
# txt4corner_file = "F:/ZYY/chess/images/train/NOCUTS_0413/txt4corner/"
# txt4corner_save = "F:/ZYY/chess/images/train/txt4corner/"

image_file = 'F:/ZYY/chess/images/test3/pre-images/gray/'
image_save = 'F:/ZYY/chess/images/test3/pre-images/gray-cut/'
txt4corner_file = 'F:/ZYY/chess/images/test3/txt4corner/txt4corner-gray/'
txt4corner_save = 'F:/ZYY/chess/images/test3/txt4corner/txt4corner-graycut/'

def corner4(f_read):
    # print(f_read)
    with open(f_read, "r") as f:
        a = f.read().split('\t')  # 读取文件
    point4 = np.zeros(8)
    for i in range(8):
        point4[i] = int(a[i])
    point4 = np.reshape(point4, (4, 2))
    return point4

path=os.path.join(image_file)
img_list=os.listdir(path)
for fileName in img_list:
    print(fileName)
    # 读图像
    img_path = image_file + fileName
    # img_path = image_file + '000414.jpg'
    img = cv2.imread(img_path)
    height, width, c = img.shape
    # 读四角坐标
    f_read = txt4corner_file + fileName.split('.')[0] + ".txt"
    point4 = corner4(f_read)
    # print(img.shape)

    # 开始剪切
    if width == height:
        y0 = 0
        y1 = height
        x0 = 0
        x1 = height
    elif width > height:
        axis = (min(point4[:, 0]) + max(point4[:, 0])) / 2
        y0 = 0
        y1 = height
        x0 =  int(axis - height / 2)
        x1 =  int(axis + height / 2)
        if x0 < 0:
            x0 = 0
            x1 = height
        for i in range(4):
            point4[i][0] -= x0
    else:
        axis = (min(point4[:, 1]) + max(point4[:, 1])) / 2
        y0 = int(axis - width / 2)
        y1 = int(axis + width / 2)
        x0 = 0
        x1 = width
        for i in range(4):
            point4[i][1] -= y0

    cropped = img[y0:y1, x0:x1]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(image_save + fileName , cropped)

    # 保存新的四角坐标
    f_save = txt4corner_save + fileName.split('.')[0] + ".txt"
    line = str(int(point4[0][0])) + '\t' + str(int(point4[0][1])) + '\t' + str(int(point4[1][0])) + '\t' + str(int(point4[1][1])) + '\t' + str(int(point4[2][0])) + '\t' + str(int(point4[2][1]))+ '\t' + str(int(point4[3][0])) + '\t' + str(int(point4[3][1]))
    with open(f_save, "w") as f:
        f.write(line)
    f.close()