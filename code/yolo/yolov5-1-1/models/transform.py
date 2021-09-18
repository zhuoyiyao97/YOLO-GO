import numpy as np
import cv2
import os

# 通过透视变换矫正棋盘形变
# img_path 图片路径， corner4四个角坐标
def imgTransform(save_dir, img_dir, img_name, corner4):
    img_path = img_dir + img_name
    print(img_path)
    img = cv2.imread(img_path)
    w, h, c= img.shape
    # point4 规则棋盘角坐标，左上 右上 左下 右下
    src = np.array([corner4[0], corner4[1], corner4[2], corner4[3]], np.float32)
    x = int(max(corner4[3][0] - corner4[0][0], corner4[3][1] - corner4[0][1]))
    x0 = int(corner4[0][0])
    y0 = int(corner4[0][1])
    new_point4 = [[x0, y0], [x0+x, y0], [x0, y0+x], [x0+x, y0+x]]
    dst = np.array(new_point4, np.float32)
    # 求映射矩阵
    M = cv2.getPerspectiveTransform(src, dst)

    # 透视变换并显示结果
    if x0 + x + 100 > w:
        w = int(x0 + x) + 100
    if y0 + x + 100 > h:
        h = int(y0 + x) + 100

    w = max(w, h)
    # 输出图像是正方形
    image = cv2.warpPerspective(img, M, (w,w))
    cropped = image[max(y0-100, 0):y0+x+100, max(x0-100, 0):x0+x+100]
    print(img_name + "  success!")
    cv2.imwrite(save_dir + 'transform/'+ img_name, cropped)

    # 保存四角坐标
    txt_save = save_dir + 'corner/' + img_name.split('.')[0] + '.txt'
    line = str(new_point4[0][0]) + '\t' + str(new_point4[0][1]) + '\t' + str(new_point4[1][0]) + '\t' + str(new_point4[1][1]) + '\t' \
           + str(new_point4[2][0]) + '\t' + str(new_point4[2][1]) + '\t' + str(new_point4[3][0]) + '\t' + str(new_point4[3][1]) + '\t'
    with open(txt_save, "w") as f:
        f.write(line)
    f.close()

def readCorner4(corner4_path):
    with open(corner4_path, "r") as f:
        a = f.read().split('\t')
    corner4 = np.zeros(8)
    for i in range(8):   
        corner4[i] = int(a[i])
    corner4 = np.reshape(corner4, (4, 2))
    sumx, sumy = 0, 0
    for i in range(4):
        sumx += corner4[i][0]
        sumy += corner4[i][1]
    centerx = sumx / 4
    centery = sumy / 4
    new = np.zeros(8).reshape(4, 2)
    # 四点坐标排序
    for i in range(4):
        if corner4[i][0] < centerx and corner4[i][1] < centery:
            new[0] = corner4[i]
        elif corner4[i][0] > centerx and corner4[i][1] < centery:
            new[1] = corner4[i]
        elif corner4[i][0] < centerx and corner4[i][1] > centery:
            new[2] = corner4[i]
        elif corner4[i][0] > centerx and corner4[i][1] > centery:
            new[3] = corner4[i]
    return new

if __name__ == '__main__':
    image_dir = 'F:/ZYY/chess/images/test3/pre-images/gray/'
    save_dir = 'F:/ZYY/chess/images/test3/'
    corne4_dir = 'F:/ZYY/chess/images/test3/txt4corner/txt4corner-gray/'
    for image in os.listdir(image_dir):
        # 读四个角坐标
        corner4_path = corne4_dir + image.split('.')[0] + ".txt"
        corner4 = readCorner4(corner4_path)
        imgTransform(save_dir, image_dir, image, corner4)








