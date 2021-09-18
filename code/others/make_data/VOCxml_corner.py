# 2 (❤ ω ❤)
# 辅助标定四个角点
# -*- coding:UTF-8 -*-
import os, sys
import glob
from PIL import Image

# 图像存储位置
src_img_dir = "F:/ZYY/chess/images/test3/pre-images/transform/"
# 图像的 ground truth 的 txt 文件存放位置
src_txt_dir = 'F:/ZYY/chess/images/test3/txt4corner/txt-trans/'
src_xml_dir = 'F:/ZYY/chess/images/test3/txt4corner/anno-transform/'

img_Lists = glob.glob(src_img_dir + '/*.jpg')

img_basenames = []  # e.g. 100.jpg
for item in img_Lists:
    img_basenames.append(os.path.basename(item))

img_names = []  # e.g. 100
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)

for img in img_names:
    f = src_img_dir + '/' + img + '.jpg'
    im = Image.open(f)
    width, height = im.size

    # open the crospronding txt file
    gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    # gt_width, gt_height = gt.size
    xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>test</folder>\n')
    xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
    xml_file.write('    <path>' + src_img_dir + '/' + str(img) + '.jpg' + '</path>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    # write the region of image on xml file
    i = 1
    for img_each_label in gt:
        spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
        imax = len(spt)
        spt[0] = "corner"
        x = int((int(spt[1]) + int(spt[3])) / 2)
        y = int((int(spt[2]) + int(spt[4])) / 2)
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + spt[0] + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(x) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(max(0, y - 10)) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(x) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(min(height, y + 10)) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')