# 2 (❤ ω ❤)
# -*- coding:UTF-8 -*-
import os, sys
import glob
from PIL import Image
def generate(src_img_dir, src_txt_dir, src_xml_dir):
    img_Lists = glob.glob(src_img_dir + '/*.jpg')

    img_basenames = []  # e.g. 100.jpg
    for item in img_Lists:
        img_basenames.append(os.path.basename(item))

    img_names = []  # e.g. 100
    for item in img_basenames:
        temp1, temp2 = os.path.splitext(item)
        img_names.append(temp1)

    for img in img_names:
        print(img)
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
            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + spt[0] + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(max(0, int(spt[1]))) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(max(0, int(spt[2]))) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(min(width, int(spt[3]))) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(min(height, int(spt[4]))) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
        xml_file.write('</annotation>')

# 图像存储位置
src_img_dir = "F:/ZYY/chess/yolov5/yolov5-1-1/inference/images/gray_images414-546/"
# 图像的 ground truth 的 txt 文件存放位置
src_txt_dir = "F:/ZYY/chess/yolov5/yolov5-1-1/inference/output/gray414-546-txt/"
src_xml_dir = "F:/ZYY/chess/yolov5/yolov5-1-1/inference/output/gray414-546-xml/"
generate(src_img_dir, src_txt_dir, src_xml_dir)