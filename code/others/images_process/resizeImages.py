import os
import cv2
''' 设置图片路径，该路径下包含了14张jpg格式的照片，名字依次为0.jpg, 1.jpg, 2.jpg,...,14.jpg'''
DATADIR="F:\ZYY\chess\images\images414-474"
'''设置目标像素大小，此处设为300'''
IMG_SIZE=640
'''使用os.path模块的join方法生成路径'''
path=os.path.join(DATADIR)

img_list=os.listdir(path)
ind=0
for i in img_list:
    '''调用cv2.imread读入图片，读入格式为IMREAD_COLOR'''
    img_array=cv2.imread(os.path.join(path,i),cv2.IMREAD_COLOR)
    '''调用cv2.resize函数resize图片'''
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    img_name=str(ind)+'.jpg'
    '''生成图片存储的目标路径'''
    save_path="F:/ZYY/chess/images/images414-474-resize/" + str(ind) + '.jpg'
    ind=ind+1
    '''调用cv.2的imwrite函数保存图片'''
    cv2.imwrite(save_path,new_array)